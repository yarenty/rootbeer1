
package org.trifort.rootbeer.examples.arraymult;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class ArrayMultApp {

  public void multArray(int[] array){
    Kernel task = new ArrayMult(array);
    Rootbeer rootbeer = new Rootbeer();
    Context context = rootbeer.createDefaultContext();
    context.init(256);
    ThreadConfig thread_config = new ThreadConfig(array.length, 1, array.length);
    Stopwatch watch = new Stopwatch();
    watch.start();
    rootbeer.run(task, thread_config, context);
    watch.stop();
    System.out.println("time: "+watch.elapsedTimeMillis());
  }
  
  public static void main(String[] args){
    ArrayMultApp app = new ArrayMultApp();
    int length = 10;
    int[] array = new int[length];
    for(int i = 0; i < array.length; ++i){
      array[i] = i;
    }
    for(int i = 0; i < array.length; ++i){
      System.out.println("start array["+i+"]: "+array[i]);
    }
    
    app.multArray(array);
    for(int i = 0; i < array.length; ++i){
      System.out.println("final array["+i+"]: "+array[i]);
    }
  }
}
