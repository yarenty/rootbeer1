/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime.nativecpu;

import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Serializer;
import org.trifort.rootbeer.runtime.gpu.GcHeap;
import org.trifort.rootbeer.runtime.gpu.GpuDevice;
import org.trifort.rootbeer.runtime.memory.BasicMemory;
import org.trifort.rootbeer.runtime.memory.BasicSwappedMemory;
import org.trifort.rootbeer.runtime.memory.BasicUnswappedMemory;
import org.trifort.rootbeer.runtime.memory.Memory;


public class NativeCpuGcHeap extends GcHeap {

  public NativeCpuGcHeap(GpuDevice device){
    super(device);  
    allocateMemory();
  }
  
  @Override
  protected void allocateMemory() {
    mBufferSize = 32*1024*1024L;
    mToSpaceMemory = new BasicSwappedMemory(512L*1024L*1024L);
    mTextureMemory = new BasicSwappedMemory(mBufferSize);
    mHandlesMemory = new BasicSwappedMemory(mBufferSize);
    mHeapEndPtrMemory = new BasicSwappedMemory(8);
    mGcInfoSpaceMemory = new BasicSwappedMemory(mGcInfoSpaceSize);
    mExceptionsMemory = new BasicSwappedMemory(mBufferSize);
  }

  @Override
  protected void makeSureReadyForUsingGarbageCollector() {
    
  }
 
  public List<Memory> getMemory(){
    List<Memory> ret = new ArrayList<Memory>();
    ret.add(mToSpaceMemory);
    ret.add(mHandlesMemory);
    ret.add(mHeapEndPtrMemory);
    ret.add(mGcInfoSpaceMemory);
    ret.add(mExceptionsMemory);
    return ret;
  }

  public Serializer getSerializer() {
    return mGcObjectVisitor;
  }
}