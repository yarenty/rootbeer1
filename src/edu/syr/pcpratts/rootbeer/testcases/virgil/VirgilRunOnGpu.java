/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.testcases.virgil;

import edu.syr.pcpratts.rootbeer.runtime.Kernel;

class VirgilRunOnGpu implements Kernel {

  private VirgilSynch m_synch;
  
  public VirgilRunOnGpu(VirgilSynch synch) {
    m_synch = synch;
  }

  public void gpuMethod() {
    
    
  }
  
  

}
