package org.trifort.rootbeer.examples.multigpu;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.util.Stopwatch;
import org.trifort.rootbeer.runtime.StatsRow;
import org.trifort.rootbeer.runtime.ThreadConfig;

public class MultiGpuApp {

  public void multArray(int[] array1, int[] array2){
    Kernel work0 = new ArrayMult(array1);
    Kernel work1 = new ArrayMult(array2);

    Rootbeer rootbeer = new Rootbeer();
    List<GpuDevice> devices = rootbeer.getDevices();
    if(devices.size() >= 2){
      System.out.println("device count: "+devices.size());
      for(GpuDevice device : devices){
        System.out.println("  name: "+device.getDeviceName());
      }
      Stopwatch watch = new Stopwatch();
      watch.start();
      GpuDevice device0 = devices.get(0);
      GpuDevice device1 = devices.get(1); 
      Context context0 = device0.createContext(4096);
      Context context1 = device1.createContext(4096);

      ThreadConfig config0 = new ThreadConfig(1, array1.length, array1.length);
      ThreadConfig config1 = new ThreadConfig(1, array2.length, array2.length);

      try {
        rootbeer.run(work0, config0, context0);
      } catch(Exception ex){
        ex.printStackTrace();
      }
      try {
        rootbeer.run(work1, config1, context1);
      } catch(Exception ex){
        ex.printStackTrace();
      }

      watch.stop();
      System.out.println("time: "+watch.elapsedTimeMillis());
    } else {
      System.out.println("This example needs two gpu devices");
      System.out.println("device count: "+devices.size());
      for(GpuDevice device : devices){
        System.out.println("  name: "+device.getDeviceName());
      }
    } 
  }
  
  public static void main(String[] args){
    MultiGpuApp app = new MultiGpuApp();
    int[] array1 = new int[5];
    int[] array2 = new int[5];
    for(int i = 0; i < array1.length; ++i){
      array1[i] = i+1;
      array2[i] = i+1;
    }
    for(int i = 0; i < array1.length; ++i){
      System.out.println("start arrays["+i+"]: "+array1[i]+" "+array2[i]);
    }
    
    app.multArray(array1, array2);
    for(int i = 0; i < array1.length; ++i){
      System.out.println("final arrays["+i+"]: "+array1[i]+" "+array2[i]);
    }
  }
}
