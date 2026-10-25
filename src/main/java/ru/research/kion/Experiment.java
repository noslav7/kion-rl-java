package ru.research.kion;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/** java -jar target/kion-rl-java-1.0.0.jar --data data/interactions.csv.zip [--fast] */
public final class Experiment {
    private Experiment() {}
    record Config(Path data,Path output,boolean demo,int users,int items,int maxTrain,int maxEval,
                  int context,int dim,int heads,int epochs,int rlSteps,long seed){}
    private static Config arguments(String[] args){
        Map<String,String> p=new HashMap<>();
        for(int i=0;i<args.length;i++){
            if(!args[i].startsWith("--"))throw new IllegalArgumentException("Unknown argument: "+args[i]);
            if(i+1<args.length&&!args[i+1].startsWith("--"))p.put(args[i++],args[i]);
            else p.put(args[i],"true");
        }
        boolean demo=p.containsKey("--demo"), medium=p.containsKey("--medium");
        return new Config(Path.of(p.getOrDefault("--data","data/kion-data.zip")),
            Path.of(p.getOrDefault("--output",demo?"outputs/demo":"outputs/kion")),demo,
            Integer.parseInt(p.getOrDefault("--max-users",medium?"5000":"1800")),
            Integer.parseInt(p.getOrDefault("--max-items",medium?"300":"160")),
            Integer.parseInt(p.getOrDefault("--max-train",medium?"2000":"650")),
            Integer.parseInt(p.getOrDefault("--max-eval",medium?"350":"120")),
            Integer.parseInt(p.getOrDefault("--context",medium?"8":"6")),
            Integer.parseInt(p.getOrDefault("--dim",medium?"12":"8")),
            Integer.parseInt(p.getOrDefault("--heads","2")),
            Integer.parseInt(p.getOrDefault("--sft-epochs",medium?"3":"2")),
            Integer.parseInt(p.getOrDefault("--rl-steps",medium?"500":"150")),
            Long.parseLong(p.getOrDefault("--seed","42")));
    }
    public static void main(String[] args)throws Exception{
        Config c=arguments(args);long started=System.nanoTime();
        Files.createDirectories(c.output);
        System.out.println("Java-only GPT-2-style sequential recommendation; mode: "+(c.demo?"SYNTHETIC DEMO — NOT KION RESULTS":"REAL KION"));
        KionDataset ds;
        if(c.demo)ds=KionDataset.demo(c.context,c.seed);
        else {
            Path input=c.data;
            if(!Files.exists(input))input=KionDataset.download(Path.of("data"));
            ds=KionDataset.load(input,c.items,c.users,c.maxTrain,c.maxEval,c.context,c.seed);
        }
        Files.writeString(c.output.resolve("run_config.txt"),"seed="+c.seed+"\nitems="+ds.items+
            "\nusers="+ds.users+"\nmax_train="+c.maxTrain+"\ncontext="+c.context+
            "\ndim="+c.dim+"\nheads="+c.heads+"\nsft_epochs="+c.epochs+"\nrl_steps="+c.rlSteps+"\n");
        Gpt2Recommender sft=new Gpt2Recommender(ds.vocab,c.context,c.dim,c.heads,c.seed);
        List<KionDataset.Example> train=new ArrayList<>(ds.train);
        Random rng=new Random(c.seed);int step=0;
        for(int ep=0;ep<c.epochs;ep++){
            Collections.shuffle(train,rng);double meanLoss=0;
            for(var ex:train){
                Gpt2Recommender.Forward f=sft.forward(ex.history());
                double[] p=Gpt2Recommender.probabilities(values(f.logits()));
                double[] grad=p.clone();grad[ex.target()]-=1;
                sft.step(f,grad,.002,++step);
                meanLoss+=-Math.log(Math.max(1e-12,p[ex.target()]));
            }
            System.out.printf(Locale.US,"SFT epoch %d/%d: loss %.4f%n",ep+1,c.epochs,meanLoss/train.size());
        }
        sft.save(c.output.resolve("sft.bin"));
        RewardModel reward=new RewardModel(ds.vocab,c.seed+7);
        reward.train(train,5,c.seed+99);reward.save(c.output.resolve("reward.bin"));
        List<Metrics.Result> metrics=new ArrayList<>();Map<String,double[]> rollouts=new LinkedHashMap<>();
        metrics.add(Metrics.evaluate("SFT",sft,reward,ds.test,ds.vocab,false));
        metrics.add(Metrics.evaluate("Reward reranking",sft,reward,ds.test,ds.vocab,true));
        rollouts.put("SFT",Metrics.rollout(sft,reward,ds.test,5));
        for(String variant:List.of("RL-lowKL","RL-highKL","RL-novelty")){
            Gpt2Recommender policy=sft.copy();double beta=variant.equals("RL-lowKL")?.03:.35;
            double novelty=variant.equals("RL-novelty")?.15:0;
            Random draw=new Random(c.seed+variant.hashCode());double baseline=0.5;
            for(int iter=0;iter<c.rlSteps;iter++){
                KionDataset.Example ex=train.get(draw.nextInt(train.size()));
                Gpt2Recommender.Forward f=policy.forward(ex.history());
                double[] p=Gpt2Recommender.probabilities(values(f.logits()));
                double[] q=Gpt2Recommender.probabilities(sft.logits(ex.history()));
                int sampled=sample(p,draw);
                double predicted=reward.predict(ex.history(),sampled);
                boolean seen=false;for(int x:ex.history())if(x==sampled){seen=true;break;}
                double r=predicted+(seen?0:novelty);
                baseline=.98*baseline+.02*r;
                double advantage=Math.max(-1,Math.min(1,r-baseline));
                double kl=0;
                for(int i=2;i<p.length;i++)if(p[i]>0)kl+=p[i]*Math.log(p[i]/Math.max(1e-12,q[i]));
                double[] gradient=new double[p.length];
                for(int i=2;i<p.length;i++){
                    // REINFORCE: -adv*grad log P(action); exact KL gradient; CE anchor.
                    double logratio=Math.log(Math.max(1e-12,p[i])/Math.max(1e-12,q[i]));
                    gradient[i]=advantage*(p[i]-(i==sampled?1:0))
                        +beta*p[i]*(logratio-kl)
                        +.20*(p[i]-(i==ex.target()?1:0));
                }
                policy.step(f,gradient,.001,iter+1);
                if((iter+1)%Math.max(1,c.rlSteps/3)==0)
                    System.out.printf(Locale.US,"%s step %d/%d: R=%.4f KL=%.4f%n",variant,iter+1,c.rlSteps,r,kl);
            }
            policy.save(c.output.resolve(variant+".bin"));
            metrics.add(Metrics.evaluate(variant,policy,reward,ds.test,ds.vocab,false));
            rollouts.put(variant,Metrics.rollout(policy,reward,ds.test,5));
        }
        Metrics.export(c.output,metrics,reward,ds,c.demo,rollouts);
        System.out.println("\npolicy, Recall@10, NDCG@10, estimated reward, coverage, max top1 share, repeat rate");
        for(Metrics.Result m:metrics)System.out.println(m.csv());
        System.out.printf(Locale.US,"%nCompleted in %.1f seconds; output: %s%n",(System.nanoTime()-started)/1e9,c.output.toAbsolutePath());
        if(c.demo)System.out.println("REMINDER: demo is synthetic; not suitable for scientific submission.");
    }
    static double[] values(AutoGrad.Value[] a){double[] out=new double[a.length];for(int i=0;i<a.length;i++)out[i]=a[i].x;return out;}
    static int sample(double[] p,Random rng){double u=rng.nextDouble();for(int i=2;i<p.length;i++){u-=p[i];if(u<=0)return i;}return p.length-1;}
}
