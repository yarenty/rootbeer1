/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.virgil;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.test.TestSerialization;
import java.util.ArrayList;
import java.util.List;

public class VirgilTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    VirgilSynch synch = new VirgilSynch();
    for(int i = 0; i < 10; ++i){
      ret.add(new VirgilRunOnGpu(synch));
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

}
