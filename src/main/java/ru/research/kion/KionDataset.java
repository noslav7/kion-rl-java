package ru.research.kion;

import java.io.*;
import java.net.*;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.function.Consumer;
import java.util.zip.*;

/** Three streaming CSV passes: global time boundaries; train-only vocabulary/users;
 *  sampled user interactions and strictly earlier-day histories. */
final class KionDataset {
    record Example(int user,int[] history,int target,double reward,int day) {}
    record Row(int user,int item,int day,double reward) {}
    final List<Example> train,val,test;
    final int vocab,excludedUnknown,context,users,items;
    final String validationStart,testStart;
    KionDataset(List<Example> train,List<Example> val,List<Example> test,int vocab,
                int excludedUnknown,int context,int users,int items,String valStart,String testStart){
        this.train=train;this.val=val;this.test=test;this.vocab=vocab;this.excludedUnknown=excludedUnknown;
        this.context=context;this.users=users;this.items=items;
        this.validationStart=valStart;this.testStart=testStart;
    }
    private static int day(String value){return (int)LocalDate.parse(value.substring(0,10)).toEpochDay();}
    static Path download(Path directory) throws IOException, InterruptedException {
        Files.createDirectories(directory);Path path=directory.resolve("kion-data.zip");
        if(Files.exists(path)&&Files.size(path)>1_000_000)return path;
        String[] urls={
            "https://raw.githubusercontent.com/irsafilo/KION_DATASET/master/interactions.csv.zip",
            "https://github.com/irsafilo/KION_DATASET/raw/master/data_original.zip"
        };
        for(String url:urls){
            try {
                System.out.println("Downloading official KION: "+url);
                HttpURLConnection c=(HttpURLConnection)new URL(url).openConnection();
                c.setConnectTimeout(15000);c.setReadTimeout(120000);c.setRequestProperty("User-Agent","kion-java-research/1.0");
                if(c.getResponseCode()!=200)throw new IOException("HTTP "+c.getResponseCode());
                try(InputStream in=c.getInputStream()){Files.copy(in,path,StandardCopyOption.REPLACE_EXISTING);}
                try(ZipFile z=new ZipFile(path.toFile())){
                    boolean ok=z.stream().anyMatch(e->e.getName().endsWith("interactions.csv"));
                    if(!ok)throw new IOException("No interactions.csv in archive");
                }
                return path;
            }catch(Exception ex){System.err.println("Download fallback: "+ex.getMessage());Files.deleteIfExists(path);}
        }
        throw new IOException("KION download failed; manually get interactions.csv.zip from official GitHub and run --data path/to/interactions.csv.zip");
    }
    private static void each(Path path, Consumer<Row> sink)throws IOException {
        if(!Files.isRegularFile(path))throw new FileNotFoundException(path.toString());
        if(path.toString().endsWith(".zip")){
            try(ZipFile zip=new ZipFile(path.toFile())){
                ZipEntry e=zip.stream().filter(x->x.getName().endsWith("interactions.csv")).findFirst()
                    .orElseThrow(()->new IOException("Archive does not contain interactions.csv"));
                try(BufferedReader br=new BufferedReader(new InputStreamReader(zip.getInputStream(e)))){parse(br,sink);}
            }
        }else try(BufferedReader br=Files.newBufferedReader(path)){parse(br,sink);}
    }
    private static void parse(BufferedReader br,Consumer<Row> sink)throws IOException{
        String header=br.readLine();if(header==null)throw new IOException("Empty CSV");
        String[] h=header.replace("\"","").split(",",-1);
        int iu=-1,ii=-1,id=-1,ip=-1;
        for(int j=0;j<h.length;j++)switch(h[j].trim().replace("\uFEFF","")){
            case "user_id"->iu=j;case "item_id"->ii=j;case "last_watch_dt"->id=j;case "watched_pct"->ip=j;
        }
        if(iu<0||ii<0||id<0||ip<0)throw new IOException("Missing required columns user_id,item_id,last_watch_dt,watched_pct");
        final int U=iu,I=ii,D=id,P=ip,max=Math.max(Math.max(iu,ii),Math.max(id,ip));
        String line;long row=0;
        while((line=br.readLine())!=null){
            row++;if(line.isEmpty())continue;String[] p=line.split(",",-1);if(p.length<=max)continue;
            try{
                int user=Integer.parseInt(p[U]),item=Integer.parseInt(p[I]),dt=day(p[D]);
                double r=p[P].isEmpty()?Double.NaN:Math.max(0,Math.min(1,Double.parseDouble(p[P])/100.0));
                sink.accept(new Row(user,item,dt,r));
            }catch(RuntimeException ignored){/* Malformed rows are dropped, never imputed from future. */}
        }
        System.out.printf("Read %,d CSV records from input stream%n",row);
    }
    static KionDataset load(Path path,int maxItems,int maxUsers,int maxTrain,int maxEval,int context,long seed)throws IOException {
        System.out.println("Pass 1/3: detecting global last calendar day...");
        int[] last={Integer.MIN_VALUE};each(path,r->last[0]=Math.max(last[0],r.day()));
        if(last[0]==Integer.MIN_VALUE)throw new IOException("No usable interactions");
        int testStart=last[0]-6,valStart=testStart-7;
        System.out.println("Chronological cutoffs: train < "+LocalDate.ofEpochDay(valStart)+
            ", validation < "+LocalDate.ofEpochDay(testStart)+", test through "+LocalDate.ofEpochDay(last[0]));
        Map<Integer,Integer> freq=new HashMap<>(),count=new HashMap<>(),minDay=new HashMap<>(),maxDay=new HashMap<>();
        System.out.println("Pass 2/3: train-only item catalog and eligible users...");
        each(path,r->{if(r.day()>=valStart)return;
            freq.merge(r.item(),1,Integer::sum);count.merge(r.user(),1,Integer::sum);
            minDay.merge(r.user(),r.day(),Math::min);maxDay.merge(r.user(),r.day(),Math::max);
        });
        List<Integer> top=freq.entrySet().stream().sorted(Comparator.<Map.Entry<Integer,Integer>>comparingInt(Map.Entry::getValue)
            .reversed().thenComparingInt(Map.Entry::getKey)).limit(maxItems).map(Map.Entry::getKey).toList();
        Map<Integer,Integer> vocab=new HashMap<>();for(int i=0;i<top.size();i++)vocab.put(top.get(i),i+2);
        List<Integer> eligible=new ArrayList<>();for(var e:count.entrySet()){
            int u=e.getKey();if(e.getValue()>=3 && maxDay.get(u)>minDay.get(u))eligible.add(u);
        }
        Collections.sort(eligible);Collections.shuffle(eligible,new Random(seed));
        Set<Integer> selected=new HashSet<>(eligible.subList(0,Math.min(maxUsers,eligible.size())));
        System.out.println("Pass 3/3: select user histories, group by exact day...");
        Map<Integer,List<Row>> byUser=new HashMap<>();
        each(path,r->{if(selected.contains(r.user()))byUser.computeIfAbsent(r.user(),k->new ArrayList<>()).add(r);});
        ArrayList<Example> train=new ArrayList<>(),val=new ArrayList<>(),test=new ArrayList<>();
        int[] unknown={0};
        for(var entry:byUser.entrySet()){
            List<Row> rows=entry.getValue();rows.sort(Comparator.comparingInt(Row::day).thenComparingInt(Row::item));
            ArrayDeque<Integer> previous=new ArrayDeque<>();int i=0;
            while(i<rows.size()){
                int j=i+1;while(j<rows.size()&&rows.get(j).day()==rows.get(i).day())j++;
                int[] h=previous.stream().mapToInt(Integer::intValue).toArray();
                for(int k=i;k<j;k++){
                    Row r=rows.get(k);Integer target=vocab.get(r.item());
                    if(target==null){unknown[0]++;continue;}
                    if(h.length==0)continue;
                    Example example=new Example(r.user(),h,target,r.reward(),r.day());
                    if(r.day()<valStart)train.add(example);
                    else if(r.day()<testStart)val.add(example);
                    else test.add(example);
                }
                // Events of the same calendar day enter the context only after its targets are formed.
                for(int k=i;k<j;k++){
                    previous.addLast(vocab.getOrDefault(rows.get(k).item(),1));
                    while(previous.size()>context)previous.removeFirst();
                }
                i=j;
            }
        }
        Random sampleRng=new Random(seed+10);
        train=sample(train,maxTrain,sampleRng);
        val=onePerUser(val,maxEval,sampleRng);
        test=onePerUser(test,maxEval,sampleRng);
        System.out.printf("Prepared train/val/test=%d/%d/%d; vocab=%d; sampled users=%d; filtered OOV=%d%n",
            train.size(),val.size(),test.size(),vocab.size()+2,selected.size(),unknown[0]);
        if(train.isEmpty()||val.isEmpty()||test.isEmpty())throw new IOException("Empty split! Try --max-items 500 --max-users 6000, or inspect dates and source CSV.");
        return new KionDataset(train,val,test,vocab.size()+2,unknown[0],context,selected.size(),vocab.size(),
            LocalDate.ofEpochDay(valStart).toString(),LocalDate.ofEpochDay(testStart).toString());
    }
    private static ArrayList<Example> onePerUser(List<Example> input,int maximum,Random r){
        Map<Integer,Example> by=new HashMap<>();
        input.stream().sorted(Comparator.comparingInt(Example::day).thenComparingInt(Example::target)).forEach(e->by.put(e.user(),e));
        return sample(new ArrayList<>(by.values()),maximum,r);
    }
    private static ArrayList<Example> sample(List<Example> input,int n,Random r){
        ArrayList<Example> copy=new ArrayList<>(input);Collections.shuffle(copy,r);
        if(copy.size()>n)copy.subList(n,copy.size()).clear();return copy;
    }
    static KionDataset demo(int context,long seed){
        Random r=new Random(seed);List<Example> tr=new ArrayList<>(),va=new ArrayList<>(),te=new ArrayList<>();
        int users=120,items=36;
        for(int u=0;u<users;u++){
            ArrayDeque<Integer> history=new ArrayDeque<>();int group=u%4;
            for(int d=0;d<30;d++){
                int[] h=history.stream().mapToInt(Integer::intValue).toArray();
                int item=2+group*9+r.nextInt(9);double reward=Math.min(1,Math.max(0,.4+.05*(item%7)+r.nextGaussian()*.1));
                if(h.length>0){Example ex=new Example(u,h,item,reward,d);
                    if(d<22)tr.add(ex);else if(d<26)va.add(ex);else te.add(ex);
                }
                history.addLast(item);while(history.size()>context)history.removeFirst();
            }
        }
        return new KionDataset(sample(tr,240,new Random(44)),onePerUser(va,90,new Random(44)),
            onePerUser(te,90,new Random(44)),items+2,0,context,users,items,"synthetic-22","synthetic-26");
    }
}
