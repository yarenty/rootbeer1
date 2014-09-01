package org.trifort.rootbeer.runtime;

public class GpuFuture {
  
  private volatile boolean ready;
  private volatile Exception ex;
  
  public GpuFuture(){
    ready = false;
  }
  
  public void signal() {
    ready = true;
  }

  public void reset() {
    ex = null;
    ready = false;
  }

  public void take() {
    while(!ready){
      //do nothing
    }
    if(ex != null){
      throw new RuntimeException(ex);
    }
  }

  public void setException(Exception ex) {
    this.ex = ex;
  }
}
