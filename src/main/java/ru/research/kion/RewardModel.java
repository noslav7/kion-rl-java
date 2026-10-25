package ru.research.kion;

import java.util.*;
import java.io.*;
import java.nio.file.*;

/** Explicit Java latent-factor reward regressor. Predicts observed watch-completion
 * fraction from history-averaged embeddings and the proposed item embedding.
 * It cannot causally evaluate unobserved user-item outcomes. */
final class RewardModel {
    private final int vocab,dim=16;
    private final double[][] emb;
    private final double[] itemBias;
    private double intercept;
    RewardModel(int vocab,long seed){
        this.vocab=vocab;emb=new double[vocab][dim];itemBias=new double[vocab];
        Random random=new Random(seed);for(double[] e:emb)for(int j=0;j<dim;j++)e[j]=random.nextGaussian()*.08;
    }
    private static double sigmoid(double x){return 1.0/(1+Math.exp(-Math.max(-30,Math.min(30,x))));}
    double predict(int[] hist,int item){
        double[] avg=new double[dim];int n=0;
        for(int it:hist){if(it<2)continue;n++;for(int k=0;k<dim;k++)avg[k]+=emb[it][k];}
        if(n>0)for(int k=0;k<dim;k++)avg[k]/=n;
        double dot=intercept+itemBias[item];for(int k=0;k<dim;k++)dot+=avg[k]*emb[item][k]/Math.sqrt(dim);
        return sigmoid(dot);
    }
    void train(List<KionDataset.Example> train,int epochs,long seed){
        List<KionDataset.Example> known=new ArrayList<>();
        for(var e:train)if(Double.isFinite(e.reward()))known.add(e);
        if(known.isEmpty())throw new IllegalArgumentException("KION contains no observed watched_pct labels");
        Random rng=new Random(seed);
        for(int ep=0;ep<epochs;ep++){
            Collections.shuffle(known,rng);double mse=0;int n=0;
            for(var e:known){
                int[] hist=e.history();int item=e.target();double[] avg=new double[dim];int count=0;
                for(int it:hist)if(it>=2){count++;for(int k=0;k<dim;k++)avg[k]+=emb[it][k];}
                if(count>0)for(int k=0;k<dim;k++)avg[k]/=count;
                double yhat=predict(hist,item),error=yhat-e.reward();
                mse+=error*error;n++;
                double dz=2*error*yhat*(1-yhat),lr=.025;
                double[] gradHistory=new double[dim];
                for(int k=0;k<dim;k++){
                    gradHistory[k]=dz*emb[item][k]/Math.sqrt(dim);
                    emb[item][k]-=lr*(dz*avg[k]/Math.sqrt(dim)+.0001*emb[item][k]);
                }
                if(count>0)for(int it:hist)if(it>=2){for(int k=0;k<dim;k++)emb[it][k]-=lr*gradHistory[k]/count;}
                intercept-=lr*dz;itemBias[item]-=lr*dz;
            }
            System.out.printf(Locale.US,"Reward epoch %d: observed train RMSE %.4f%n",ep+1,Math.sqrt(mse/Math.max(1,n)));
        }
    }
    double[] errors(List<KionDataset.Example> examples){
        int n=0;double mae=0,mse=0;
        for(var ex:examples){if(!Double.isFinite(ex.reward()))continue;
            double err=predict(ex.history(),ex.target())-ex.reward();mae+=Math.abs(err);mse+=err*err;n++;
        }
        return new double[]{mae/Math.max(1,n),Math.sqrt(mse/Math.max(1,n)),n};
    }
    void save(Path path)throws IOException{
        try(DataOutputStream out=new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))){
            out.writeUTF("REWARDJAVA1");out.writeInt(vocab);out.writeInt(dim);out.writeDouble(intercept);
            for(double[] a:emb)for(double v:a)out.writeDouble(v);
            for(double d:itemBias)out.writeDouble(d);
        }
    }
}
