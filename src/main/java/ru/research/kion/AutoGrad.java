package ru.research.kion;

import java.util.*;

/** Lightweight reverse-mode automatic differentiation. All computations and gradients are Java. */
final class AutoGrad {
    private AutoGrad() {}
    static final class Value {
        double x, grad, m, v;
        Value a, b;
        double da, db;
        Value(double x) { this.x=x; }
    }
    static final class Tape {
        final ArrayList<Value> nodes = new ArrayList<>(16384);
        Value add(Value a, Value b) { return make(a.x+b.x,a,b,1,1); }
        Value add(Value a, double x) {return make(a.x+x,a,null,1,0);}
        Value sub(Value a, Value b) {return make(a.x-b.x,a,b,1,-1);}
        Value mul(Value a, Value b) {return make(a.x*b.x,a,b,b.x,a.x);}
        Value mul(Value a, double x) {return make(a.x*x,a,null,x,0);}
        Value invSqrt(Value a) {
            double x=Math.pow(Math.max(1e-12,a.x),-0.5);
            return make(x,a,null,-0.5*x/Math.max(1e-12,a.x),0);
        }
        Value exp(Value a) {double x=Math.exp(Math.max(-50, Math.min(50,a.x)));return make(x,a,null,x,0);}
        Value gelu(Value a) {
            double x=a.x;
            double z=Math.sqrt(2.0/Math.PI)*(x+0.044715*x*x*x);
            double th=Math.tanh(z), dz=Math.sqrt(2.0/Math.PI)*(1+3*0.044715*x*x);
            return make(0.5*x*(1+th),a,null,0.5*(1+th)+0.5*x*(1-th*th)*dz,0);
        }
        private Value make(double x, Value a, Value b, double da, double db) {
            Value out=new Value(x);out.a=a;out.b=b;out.da=da;out.db=db;nodes.add(out);return out;
        }
        Value sum(Value[] values) {
            if(values.length==0)throw new IllegalArgumentException("Empty sum");
            Value out=values[0];for(int i=1;i<values.length;i++)out=add(out,values[i]);return out;
        }
        Value dot(Value[] a, Value[] b) {
            Value[] parts=new Value[a.length];for(int i=0;i<a.length;i++)parts[i]=mul(a[i],b[i]);return sum(parts);
        }
        Value[] softmax(Value[] logits) {
            double max=Double.NEGATIVE_INFINITY;for(Value v:logits)max=Math.max(max,v.x);
            Value[] ex=new Value[logits.length];for(int i=0;i<ex.length;i++)ex[i]=exp(add(logits[i],-max));
            Value total=sum(ex), inv=inv(total);
            Value[] out=new Value[ex.length];for(int i=0;i<out.length;i++)out[i]=mul(ex[i],inv);return out;
        }
        Value inv(Value a) {
            double x=1.0/a.x;return make(x,a,null,-x*x,0);
        }
        void backward(Value[] logits, double[] dlogits) {
            if(logits.length!=dlogits.length)throw new IllegalArgumentException("Output gradient shape");
            for(int i=0;i<logits.length;i++)logits[i].grad+=dlogits[i];
            for(int i=nodes.size()-1;i>=0;i--){
                Value n=nodes.get(i);
                if(n.a!=null)n.a.grad+=n.grad*n.da;
                if(n.b!=null)n.b.grad+=n.grad*n.db;
            }
        }
    }
    static void adam(List<Value> params, double lr, int step, double clip) {
        double norm=0;for(Value p:params){norm+=p.grad*p.grad;}
        double scale=Math.min(1.0, clip/(Math.sqrt(norm)+1e-12));
        for(Value p:params){
            double g=p.grad*scale;
            p.m=0.9*p.m+0.1*g;p.v=0.999*p.v+0.001*g*g;
            double mh=p.m/(1-Math.pow(0.9,step)),vh=p.v/(1-Math.pow(0.999,step));
            p.x-=lr*mh/(Math.sqrt(vh)+1e-8);p.grad=0;
        }
    }
}
