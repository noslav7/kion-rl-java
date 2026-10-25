package ru.research.kion;

import java.util.*;

/** Validation of reverse-mode gradients using finite differences, no external test libraries. */
public final class Diagnostics {
    private Diagnostics(){}
    public static void main(String[] args){
        Gpt2Recommender m=new Gpt2Recommender(14,4,8,2,123);
        int[] h={2,3,4};int target=5;
        var f=m.forward(h);
        double[] logits=Experiment.values(f.logits());
        double[] p=Gpt2Recommender.probabilities(logits);
        double[] g=p.clone();g[target]-=1;
        f.tape().backward(f.logits(),g);
        int[] indices={0,9,51,m.parameters.size()/2,m.parameters.size()-1};
        double eps=1e-5;
        for(int idx:indices){
            var w=m.parameters.get(idx);double old=w.x,analytic=w.grad;
            w.x=old+eps;double plus=loss(m.logits(h),target);
            w.x=old-eps;double minus=loss(m.logits(h),target);
            w.x=old;
            double finite=(plus-minus)/(2*eps);
            double rel=Math.abs(analytic-finite)/Math.max(1e-5,Math.abs(analytic)+Math.abs(finite));
            if(rel>1e-3)throw new AssertionError("Gradient mismatch at "+idx+" analytic="+analytic+" numeric="+finite+" relative="+rel);
            System.out.printf(Locale.US,"Gradient parameter %d: analytic=%g, finite-diff=%g, rel-error=%g%n",idx,analytic,finite,rel);
        }
        System.out.println("AUTOGRAD CHECK PASSED");
    }
    private static double loss(double[] logits,int y){
        double[] p=Gpt2Recommender.probabilities(logits);return -Math.log(p[y]);
    }
}
