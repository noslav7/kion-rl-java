package ru.research.kion;

import java.util.*;
import static ru.research.kion.AutoGrad.*;

/** One-block GPT-2-style causal Transformer: learned embeddings, pre-LN, multi-head
 * masked self-attention, residual connections, GELU FFN, final LN, tied LM head.
 * No pretrained language weights are used. This is a deliberately tiny architecture.
 */
final class Gpt2Recommender {
    final int vocabulary, context, dim, heads, headDim;
    final ArrayList<Value> parameters=new ArrayList<>();
    final Value[][] tok, pos, qkv, proj, fc1, fc2;
    final Value[] qb, pb, f1b, f2b, ln1g,ln1b,ln2g,ln2b,lfG,lfB;
    private final Random rng;
    record Forward(Tape tape, Value[] logits) {}
    Gpt2Recommender(int vocabulary,int context,int dim,int heads,long seed) {
        if(dim%heads!=0)throw new IllegalArgumentException("dim must be multiple of heads");
        this.vocabulary=vocabulary;this.context=context;this.dim=dim;this.heads=heads;headDim=dim/heads;
        rng=new Random(seed);
        tok=matrix(vocabulary,dim,.03);pos=matrix(context,dim,.02);
        qkv=matrix(dim,3*dim,.06);proj=matrix(dim,dim,.06);
        fc1=matrix(dim,4*dim,.05);fc2=matrix(4*dim,dim,.05);
        qb=vector(3*dim,0);pb=vector(dim,0);f1b=vector(4*dim,0);f2b=vector(dim,0);
        ln1g=vector(dim,1);ln1b=vector(dim,0);ln2g=vector(dim,1);ln2b=vector(dim,0);
        lfG=vector(dim,1);lfB=vector(dim,0);
    }
    private Value[][] matrix(int rows,int cols,double stdev){
        Value[][] a=new Value[rows][cols];for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)a[i][j]=weight(rng.nextGaussian()*stdev);return a;
    }
    private Value[] vector(int n,double val){Value[] a=new Value[n];for(int i=0;i<n;i++)a[i]=weight(val);return a;}
    private Value weight(double x){Value p=new Value(x);parameters.add(p);return p;}
    private Value[] linear(Tape t, Value[] x, Value[][] w, Value[] bias) {
        Value[] out=new Value[bias.length];for(int j=0;j<out.length;j++){
            Value sum=t.add(bias[j],0);for(int i=0;i<x.length;i++)sum=t.add(sum,t.mul(x[i],w[i][j]));out[j]=sum;
        }return out;
    }
    private Value[] layerNorm(Tape t,Value[] x,Value[] gamma,Value[] beta){
        Value mean=t.mul(t.sum(x),1.0/x.length), var=t.add(mean,0);
        Value[] diff=new Value[x.length],squares=new Value[x.length];
        for(int i=0;i<x.length;i++){diff[i]=t.sub(x[i],mean);squares[i]=t.mul(diff[i],diff[i]);}
        var=t.add(t.mul(t.sum(squares),1.0/x.length),1e-5);
        Value inv=t.invSqrt(var), out[]=new Value[x.length];
        for(int i=0;i<x.length;i++)out[i]=t.add(t.mul(t.mul(diff[i],inv),gamma[i]),beta[i]);
        return out;
    }
    private Value[] add(Tape t,Value[] a,Value[] b){Value[] r=new Value[a.length];for(int i=0;i<r.length;i++)r[i]=t.add(a[i],b[i]);return r;}
    Forward forward(int[] history){
        if(history.length<1||history.length>context)throw new IllegalArgumentException("history length 1.."+context);
        Tape t=new Tape();int n=history.length;
        Value[][] x=new Value[n][];
        for(int i=0;i<n;i++){
            x[i]=new Value[dim];int token=history[i];
            for(int d=0;d<dim;d++)x[i][d]=t.add(tok[token][d],pos[i][d]);
        }
        Value[][] q=new Value[n][dim],k=new Value[n][dim],v=new Value[n][dim];
        for(int i=0;i<n;i++){
            Value[] h=linear(t,layerNorm(t,x[i],ln1g,ln1b),qkv,qb);
            System.arraycopy(h,0,q[i],0,dim);System.arraycopy(h,dim,k[i],0,dim);
            System.arraycopy(h,2*dim,v[i],0,dim);
        }
        Value[][] aOut=new Value[n][dim];
        for(int i=0;i<n;i++){
            for(int h=0;h<heads;h++){
                Value[] scores=new Value[i+1];
                for(int j=0;j<=i;j++){
                    Value[] qi=Arrays.copyOfRange(q[i],h*headDim,(h+1)*headDim);
                    Value[] kj=Arrays.copyOfRange(k[j],h*headDim,(h+1)*headDim);
                    scores[j]=t.mul(t.dot(qi,kj),1.0/Math.sqrt(headDim));
                }
                Value[] probs=t.softmax(scores);
                for(int d=0;d<headDim;d++){
                    Value sum=t.mul(v[0][h*headDim+d],probs[0]);
                    for(int j=1;j<=i;j++)sum=t.add(sum,t.mul(v[j][h*headDim+d],probs[j]));
                    aOut[i][h*headDim+d]=sum;
                }
            }
        }
        for(int i=0;i<n;i++){
            Value[] y=add(t,x[i],linear(t,aOut[i],proj,pb));
            Value[] n2=layerNorm(t,y,ln2g,ln2b),hidden=linear(t,n2,fc1,f1b);
            for(int j=0;j<hidden.length;j++)hidden[j]=t.gelu(hidden[j]);
            x[i]=add(t,y,linear(t,hidden,fc2,f2b));
        }
        Value[] last=layerNorm(t,x[n-1],lfG,lfB),logits=new Value[vocabulary];
        // Weight tying as in GPT-2; PAD=0 and UNK=1 are masked during candidate selection.
        for(int j=0;j<vocabulary;j++)logits[j]=t.dot(last,tok[j]);
        return new Forward(t,logits);
    }
    double[] logits(int[] history){Forward f=forward(history);double[] out=new double[vocabulary];for(int i=0;i<vocabulary;i++)out[i]=f.logits()[i].x;return out;}
    static double[] probabilities(double[] logits){
        double max=Double.NEGATIVE_INFINITY;for(int i=2;i<logits.length;i++)max=Math.max(max,logits[i]);
        double sum=0;double[] p=new double[logits.length];
        for(int i=2;i<logits.length;i++){p[i]=Math.exp(Math.max(-60,logits[i]-max));sum+=p[i];}
        for(int i=2;i<logits.length;i++)p[i]/=sum;return p;
    }
    void step(Forward f,double[] derivatives,double learningRate,int step){
        f.tape().backward(f.logits(),derivatives);AutoGrad.adam(parameters,learningRate,step,1.0);
    }
    void save(java.nio.file.Path file)throws java.io.IOException{
        java.nio.file.Files.createDirectories(file.toAbsolutePath().getParent());
        try(java.io.DataOutputStream out=new java.io.DataOutputStream(new java.io.BufferedOutputStream(java.nio.file.Files.newOutputStream(file)))){
            out.writeUTF("GPT2JAVA1");out.writeInt(vocabulary);out.writeInt(context);out.writeInt(dim);out.writeInt(heads);
            out.writeInt(parameters.size());for(Value p:parameters)out.writeDouble(p.x);
        }
    }
    static Gpt2Recommender load(java.nio.file.Path file)throws java.io.IOException {
        try(java.io.DataInputStream in=new java.io.DataInputStream(new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(file)))){
            if(!"GPT2JAVA1".equals(in.readUTF()))throw new java.io.IOException("Wrong checkpoint");
            Gpt2Recommender m=new Gpt2Recommender(in.readInt(),in.readInt(),in.readInt(),in.readInt(),42);
            int n=in.readInt();if(n!=m.parameters.size())throw new java.io.IOException("Parameter count mismatch");
            for(Value p:m.parameters)p.x=in.readDouble();return m;
        }
    }
    Gpt2Recommender copy(){Gpt2Recommender m=new Gpt2Recommender(vocabulary,context,dim,heads,43);for(int i=0;i<parameters.size();i++)m.parameters.get(i).x=parameters.get(i).x;return m;}
}
