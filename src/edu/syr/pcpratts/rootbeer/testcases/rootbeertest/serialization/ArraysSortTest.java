package edu.syr.pcpratts.rootbeer.testcases.rootbeertest.serialization;

import java.util.ArrayList;
import java.util.List;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;

public class ArraysSortTest implements TestSerialization {

  @Override
	public List<Kernel> create() {
	  List<Kernel> ret = new ArrayList<Kernel>();
	  for(int i = 0; i < 1; ++i){
	    ret.add(new ArraysSortRunOnGpu());
	  }
	  return ret;
  }

  @Override
  public boolean compare(Kernel original, Kernel from_heap) {
    ArraysSortRunOnGpu lhs = (ArraysSortRunOnGpu) original;
    ArraysSortRunOnGpu rhs = (ArraysSortRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
