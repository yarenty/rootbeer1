/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class ArrayLengthTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 27; ++i){
      ret.add(new ArrayLengthRunOnGpu(new int[i]));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    ArrayLengthRunOnGpu lhs = (ArrayLengthRunOnGpu) original;
    ArrayLengthRunOnGpu rhs = (ArrayLengthRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
