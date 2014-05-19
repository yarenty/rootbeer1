package org.trifort.rootbeer.testcases.rootbeertest.kerneltemplate;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

/**
 * Notes:
 *   1. array length does not go to zero. with syncthreads, the application
 *      terminates early
 *   2. without syncthreads it terminates early too
 *   3. attempting to use synchronized(this). that didn't work
 *   4. attempting to print exit thread
 *   5. printing directly after debug makes it pass
 *   6. allocating directly after debug fails
 *   7. printing a pre allocated string directly after debug makes it pass
 *   8. removing print of string but leaving it allocated: didn't pass. but result was 2
 *   9. printing number. for loop over vector.length does not print
 *   10. printing vector length with string causes pass
 *   11. printing vector length without string: causes fail
 *   12. adding volatile in primitive local declarations: fail
 * @author pcpratts
 *
 */
public class GpuVectorMapRunOnGpu2 implements Kernel {

  public GpuVectorMap2 m_map;

  public GpuVectorMapRunOnGpu2(GpuVectorMap2 map) {
    this.m_map = map;
  }

  @Override
  public void gpuMethod() {
    int thread_idxx = RootbeerGpu.getThreadIdxx();
    int block_idxx = RootbeerGpu.getBlockIdxx();
    
    String hello = "hello world";
    int number = 10;

    // Setup sharedMemory from Map
    if (thread_idxx == 0) {
      double[] vector = m_map.get(block_idxx);
      debug(block_idxx, vector); // TODO Error
      //System.out.println(hello);
      System.out.println(number);
      //System.out.println("vector.length: "+vector.length);
      for(int i = 0; i < 20; ++i){
        System.out.println(vector.length);
      }
      for (int i = 0; i < vector.length; i++) {
        System.out.println("i: "+i);
        RootbeerGpu.setSharedDouble(i * 8, vector[i]);
        System.out.println("vector[i]: "+vector[i]);
      }
    }
    
    
    RootbeerGpu.threadfenceBlock();
    
    RootbeerGpu.syncthreads();

    // Each kernel increments one item
    System.out.println("thread_idxx: "+thread_idxx);
    double val = RootbeerGpu.getSharedDouble(thread_idxx * 8);
    System.out.println("value: "+val);
    RootbeerGpu.setSharedDouble(thread_idxx * 8, val + 1);

    RootbeerGpu.syncthreads();

    // Put sharedMemory back into Map
    if (thread_idxx == 0) {
      double[] vector = new double[RootbeerGpu.getBlockDimx()];
      for (int i = 0; i < vector.length; i++) {
        System.out.println("i: "+i);
        vector[i] = RootbeerGpu.getSharedDouble(i * 8);
        System.out.println("vector[i]: "+vector[i]);
      }
      m_map.put(block_idxx, vector);
    }
  }

  //private synchronized void debug(int val, double[] arr) {
  private void debug(int val, double[] arr) {
    synchronized(this){
      System.out.println("enter thread: "+RootbeerGpu.getThreadId());
      int x = arr.length; // ERROR arr.length sets array values to 0
      System.out.println("arr.length: "+x);
    }
    System.out.println("exit thread");
    // System.out.print("(");
    // System.out.print(val);
    // System.out.print(",");
    // if (arr != null) {
    // for (int i = 0; i < arr.length; i++) {
    // System.out.print(Double.toString(arr[i]));
    // if (i + 1 < arr.length) {
    // System.out.print(",");
    // }
    // }
    // }
    // System.out.println(")");
  }
}
