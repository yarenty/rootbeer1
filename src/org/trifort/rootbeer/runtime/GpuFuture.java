package org.trifort.rootbeer.runtime;

public class GpuFuture {
  
  private volatile boolean ready;
  
  public GpuFuture(){
    ready = false;
  }
  
  public void signal() {
    ready = true;
  }

  public void reset() {
    ready = false;
  }

  public void take() {
    while(!ready){
      //do nothing
    }
  }
}
