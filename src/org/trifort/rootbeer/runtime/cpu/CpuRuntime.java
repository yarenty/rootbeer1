/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime.cpu;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.trifort.rootbeer.runtime.*;

public class CpuRuntime {

  private static CpuRuntime mInstance = null;
  private List<CpuCore> m_Cores;

  public static CpuRuntime v(){
    if(mInstance == null)
      mInstance = new CpuRuntime();
    return mInstance;
  }

  private CpuRuntime(){
    m_Cores = new ArrayList<CpuCore>();
    int num_cores = Runtime.getRuntime().availableProcessors();
    for(int i = 0; i < num_cores; ++i){
      m_Cores.add(new CpuCore());
    }
  }


  public boolean isGpuPresent() {
    return true;
  }


}
