/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.gpurequired;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class BarrierTest implements TestSerialization{

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    int len = 20;
    int[] array = new int[len];
    for(int i = 0; i < len; ++i){
      array[i] = len - i - 1;
    }
    for(int i = 0; i < len; ++i){
      ret.add(new BarrierRunOnGpu(array, i));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    BarrierRunOnGpu lhs = (BarrierRunOnGpu) original;
    BarrierRunOnGpu rhs = (BarrierRunOnGpu) from_heap;
    
    return lhs.compare(rhs);
  }

}
