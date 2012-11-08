
package rootbeer.examples.mmult;

import java.util.List;
import java.util.ArrayList;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.Rootbeer;
import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;

public class MMultApp {

  public void multMatrices(int[] a, int[] b, int[] c){
    List<Kernel> jobs = new ArrayList<Kernel>();
    for(int i = 0; i < a.length; ++i){
      jobs.add(new MMult(a, b, c, i, a.length));
    }

    Rootbeer rootbeer = new Rootbeer();
    rootbeer.runAll(jobs);
  }

  public void cpuMultMatrices(int[] a, int[] b, int[] c){
    for(int x = 0; x < a.length; ++x){
      for(int y = 0; y < a.length; ++y){
        int sum = 0;
        for(int k = 0; k < a.length; ++k){
          sum += (a[x*a.length+y]*b[y*a.length+k]);
        }
        c[x*a.length+y] = sum;
      }
    }
  }

  public static void main(String[] args){
    MMultApp app = new MMultApp();
    int size = 1024;
    int[] a = new int[size*size];
    int[] b = new int[size*size];
    int[] c_gpu = new int[size*size];
    int[] c_cpu = new int[size*size];
    
    for(int x = 0; x < size; ++x){
      for(int y = 0; y < size; ++y){
        a[x*size+y] = x*size+y;
        b[x*size+y] = x*size+y;
      }
    }

    Stopwatch watch = new Stopwatch();
    watch.start();
    app.multMatrices(a, b, c_gpu);
    watch.stop();
    System.out.println("gpu time: "+watch.elapsedTimeMillis());

    watch = new Stopwatch();
    watch.start();
    app.cpuMultMatrices(a, b, c_cpu);
    watch.stop();
    System.out.println("cpu time: "+watch.elapsedTimeMillis());
  }
}
