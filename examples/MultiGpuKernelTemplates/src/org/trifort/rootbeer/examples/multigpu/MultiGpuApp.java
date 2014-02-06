package org.trifort.rootbeer.examples.multigpu;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.util.Stopwatch;

public class MultiGpuApp {

  public void multArray(int[] array1, int[] array2){
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

      ThreadConfig config = new ThreadConfig(array1.length, 1, array1.length);
      Kernel kernel0 = (Kernel) new ArrayMult(array1);
      Kernel kernel1 = (Kernel) new ArrayMult(array2);

      rootbeer.run(kernel0, config, context0);
      //rootbeer.run(kernel1, config, context1);
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
      array1[i] = i;
      array2[i] = i;
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
