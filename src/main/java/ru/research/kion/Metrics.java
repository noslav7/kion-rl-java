package ru.research.kion;

import java.util.*;
import java.io.*;
import java.nio.file.*;

/** Identical chronological holdout evaluation for each learned policy. */
final class Metrics {
    record Result(String name,double recall10,double ndcg10,double meanPredictedReward,
                  double coverage,double maxTop1Share,double historyRepeatRate,
                  int uniqueTop1,int cases) {
        String csv(){return String.format(Locale.US,"%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%d",name,
            recall10,ndcg10,meanPredictedReward,coverage,maxTop1Share,historyRepeatRate,uniqueTop1,cases);}
    }
    static Result evaluate(String name,Gpt2Recommender model,RewardModel reward,List<KionDataset.Example> data,
                           int vocab,boolean rerank){
        int hits=0,n=0,ten=10;double ndcg=0,rewardSum=0,repeat=0;
        Map<Integer,Integer> top1Counts=new HashMap<>();Set<Integer> recommended=new HashSet<>();
        for(var ex:data){
            double[] logits=model.logits(ex.history());int[] sorted=rank(logits,rerank?Math.min(50,vocab-2):Math.min(10,vocab-2));
            if(rerank){
                // Rerank top 50 SFT candidates with reward. This is deliberately NOT policy training.
                Integer[] boxed=Arrays.stream(sorted).boxed().toArray(Integer[]::new);
                Arrays.sort(boxed,(x,y)->Double.compare(reward.predict(ex.history(),y),reward.predict(ex.history(),x)));
                sorted=Arrays.stream(boxed).mapToInt(Integer::intValue).toArray();
            }
            n++;int k=Math.min(ten,sorted.length);Set<Integer> seen=new HashSet<>();for(int x:ex.history())seen.add(x);
            for(int i=0;i<k;i++){
                int item=sorted[i];recommended.add(item);
                if(seen.contains(item))repeat+=1.0/k;
                if(item==ex.target()){hits++;ndcg+=1.0/(Math.log(i+2)/Math.log(2));}
            }
            if(k>0){top1Counts.merge(sorted[0],1,Integer::sum);rewardSum+=reward.predict(ex.history(),sorted[0]);}
        }
        int max=top1Counts.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        return new Result(name,hits/(double)Math.max(1,n),ndcg/Math.max(1,n),rewardSum/Math.max(1,n),
            recommended.size()/(double)Math.max(1,vocab-2),max/(double)Math.max(1,n),repeat/Math.max(1,n),top1Counts.size(),n);
    }
    static int[] rank(double[] logits,int k){
        PriorityQueue<Integer> minHeap=new PriorityQueue<>((a,b)->Double.compare(logits[a],logits[b]));
        for(int i=2;i<logits.length;i++){
            if(minHeap.size()<k)minHeap.add(i);
            else if(logits[i]>logits[minHeap.peek()]){minHeap.poll();minHeap.add(i);}
        }
        ArrayList<Integer> out=new ArrayList<>(minHeap);out.sort((a,b)->Double.compare(logits[b],logits[a]));
        return out.stream().mapToInt(Integer::intValue).toArray();
    }
    static double[] rollout(Gpt2Recommender model,RewardModel reward,List<KionDataset.Example> data,int horizon){
        double[] out=new double[horizon];if(data.isEmpty())return out;
        for(var ex:data){
            ArrayDeque<Integer> history=new ArrayDeque<>();for(int x:ex.history())history.add(x);
            for(int i=0;i<horizon;i++){
                int[] h=history.stream().mapToInt(Integer::intValue).toArray();
                int next=rank(model.logits(h),1)[0];out[i]+=reward.predict(h,next);
                history.addLast(next);while(history.size()>model.context)history.removeFirst();
            }
        }
        for(int i=0;i<horizon;i++)out[i]/=data.size();return out;
    }
    static void export(Path out,List<Result> rows,RewardModel reward,KionDataset ds,boolean demo,
                       Map<String,double[]> rollout)throws IOException{
        Files.createDirectories(out);
        List<String> csv=new ArrayList<>();csv.add("policy,recall_at_10,ndcg_at_10,predicted_reward_top1,catalog_coverage_at_10,max_top1_share,history_repeat_at_10,unique_top1,cases");
        for(Result r:rows)csv.add(r.csv());Files.write(out.resolve("metrics.csv"),csv);
        double[] val=reward.errors(ds.val),test=reward.errors(ds.test);
        StringBuilder report=new StringBuilder();
        report.append("# ").append(demo?"SYNTHETIC SMOKE TEST — NOT VALID KION RESULTS":"KION — Java GPT-2-style + reward model + policy-gradient RL").append("\n\n");
        if(demo)report.append("**WARNING: Only generated mock events were used. Do not submit these numerical results as evidence about KION.**\n\n");
        report.append("## Data and protocol\n\n")
          .append("- Dataset: ").append(demo?"synthetic generated events":"official MTS KION interactions.csv").append("\n")
          .append("- Dates: train before ").append(ds.validationStart).append("; validation until ").append(ds.testStart).append("; test thereafter.\n")
          .append("- Selected users: ").append(ds.users).append("; train-only candidate catalog: ").append(ds.items).append(".\n")
          .append("- Examples: train ").append(ds.train.size()).append(", validation ").append(ds.val.size()).append(", test ").append(ds.test.size()).append(".\n")
          .append("- Excluded held-out items not in train catalog (including train samples): ").append(ds.excludedUnknown).append(".\n")
          .append("- Histories include events from **strictly earlier calendar days**; no within-day order is assumed.\n")
          .append("- One target per user in each held-out split; each policy uses the same evaluation examples.\n\n")
          .append("## Reward model\n\n")
          .append(String.format(Locale.US,"Observed validation: MAE=%.5f, RMSE=%.5f, N=%d.  \n",val[0],val[1],(int)val[2]))
          .append(String.format(Locale.US,"Observed test: MAE=%.5f, RMSE=%.5f, N=%d.\n\n",test[0],test[1],(int)test[2]))
          .append("The reward model is a history-conditioned latent-factor sigmoid regressor predicting capped watched_pct/100. ")
          .append("Its predictions for novel policy choices are **not observed counterfactual user rewards**.\n\n")
          .append("## Comparison (same held-out test, no test-driven selection)\n\n")
          .append("| Policy | Recall@10 | NDCG@10 | Proxy reward | Coverage@10 | Top1 max share | Repeat@10 |\n")
          .append("|---|---:|---:|---:|---:|---:|---:|\n");
        for(Result x:rows)report.append(String.format(Locale.US,"| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n",
            x.name,x.recall10,x.ndcg10,x.meanPredictedReward,x.coverage,x.maxTop1Share,x.historyRepeatRate));
        report.append("\n## Five-step autoregressive rollout: mean *predicted* reward\n\n")
              .append("| Policy | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |\n|---|---:|---:|---:|---:|---:|\n");
        for(var e:rollout.entrySet()){
            report.append("| ").append(e.getKey()).append(" | ");for(double v:e.getValue())report.append(String.format(Locale.US,"%.4f | ",v));report.append("\n");
        }
        report.append("\n**Interpretation:** A generated recommendation is NOT a confirmed play. Appending it to a history creates ")
              .append("counterfactual states; compounding errors and reward-model extrapolation make these rollouts only a diagnostic.\n\n")
              .append("## What this experiment can and cannot conclude\n\n")
              .append("The REINFORCE runs update the actual Transformer parameters using a sampled policy gradient, an exact categorical KL term relative to frozen SFT, ")
              .append("and a supervised CE anchor. The reward model is frozen during RL. ")
              .append("Any increase in its predicted score is **not** proof of improved watch time or retention. ")
              .append("Observational logs lack randomized action propensities. Online A/B testing is required for causal conclusions. ")
              .append("Long-term objectives might use next-session return and total subsequent meaningful interactions, but these are not established by this offline experiment.\n\n")
              .append("## Limitations\n\n")
              .append("- This is a tiny **GPT-2-style** decoder (one causal pre-LN Transformer block), not official pretrained GPT-2.\n")
              .append("- CPU-only minimal Java autodiff prioritizes auditability and limited execution time; the sampled catalog makes metrics incomparable to full benchmark results.\n")
              .append("- Test is warm-catalog conditional: unseen items are excluded. Same-day interactions are unordered.\n")
              .append("- KL regularization is on the sampled history states, not a guarantee of unchanged production policy.\n")
              .append("- Training is stochastic; seed and run settings should be recorded. Repeat with multiple seeds for stronger conclusions.\n");
        Files.writeString(out.resolve("EXPERIMENT_REPORT.md"),report.toString());
    }
}
