/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime.gpu;

import java.util.Iterator;

import org.trifort.rootbeer.runtime.Kernel;

public interface GpuDevice {

  public GcHeap CreateHeap();
  public long getMaxEnqueueSize();
  public long getNumBlocks();
  public void flushQueue();   
  public long getMaxMemoryAllocSize();
  public long getGlobalMemSize();
}
