/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

import org.trifort.rootbeer.runtime.memory.Memory;

public interface CompiledKernel {
  public String getCodeUnix();
  public String getCodeWindows();
  public int getNullPointerNumber();
  public int getOutOfMemoryNumber();
  public String getCubin32();
  public String getCubin64();
  public Serializer getSerializer(GpuMemory mem);
  public boolean isUsingGarbageCollector();
}
