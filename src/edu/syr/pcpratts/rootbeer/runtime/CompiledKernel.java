/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime;

import edu.syr.pcpratts.rootbeer.runtime.memory.Memory;
import soot.Scene;
import soot.SootClass;
import soot.rbclassload.RootbeerClassLoader;

public interface CompiledKernel {
  public String getCodeUnix();
  public String getCodeWindows();
  public int getNullPointerNumber();
  public int getOutOfMemoryNumber();
  public String getCubin();
  public Serializer getSerializer(Memory mem, Memory texture_mem);
  public boolean isUsingGarbageCollector();
}
