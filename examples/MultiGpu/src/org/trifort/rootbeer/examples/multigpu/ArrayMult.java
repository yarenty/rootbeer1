package org.trifort.rootbeer.examples.multigpu;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

public class ArrayMult implements Kernel {
  
  private int[] source;
  
  public ArrayMult(int[] source){
    this.source = source;
  }
  
  public void gpuMethod(){
    int index = RootbeerGpu.getThreadId();
    source[index] *= 11;
  }
}
