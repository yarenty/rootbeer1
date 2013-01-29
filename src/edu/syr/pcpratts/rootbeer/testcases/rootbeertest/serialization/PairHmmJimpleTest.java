package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class PairHmmJimpleTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 20; ++i){
      ret.add(new PairHmmJimpleRunOnGpu());
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    PairHmmJimpleRunOnGpu lhs = (PairHmmJimpleRunOnGpu) original;
    PairHmmJimpleRunOnGpu rhs = (PairHmmJimpleRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
  
}
