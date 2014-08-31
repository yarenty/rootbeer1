package org.trifort.rootbeer.runtime;

import com.lmax.disruptor.EventFactory;

public class GpuEvent {
  private GpuEventCommand value;
  final private GpuFuture future;
  
  public GpuEvent(){
    future = new GpuFuture();
  }

  public GpuEventCommand getValue() {
    return value;
  }
  
  public GpuFuture getFuture(){
    return future;
  }

  public void setValue(GpuEventCommand value) {
    this.value = value;
  }

  public final static EventFactory<GpuEvent> EVENT_FACTORY = new EventFactory<GpuEvent>() {
    public GpuEvent newInstance() {
      return new GpuEvent();
    }
  };
}
