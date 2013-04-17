/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.gpu;

import edu.syr.pcpratts.rootbeer.entry.Aug4th2011PerformanceStudy;
import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.runtime.Serializer;
import edu.syr.pcpratts.rootbeer.runtime.PartiallyCompletedParallelJob;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.CompiledKernel;
import edu.syr.pcpratts.rootbeer.runtime.memory.Memory;
import edu.syr.pcpratts.rootbeer.runtime.memory.BufferPrinter;
import edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch;
import edu.syr.pcpratts.rootbeer.runtimegpu.GpuException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public abstract class GcHeap {
  private List<CompiledKernel> mBlocks;

  protected final int mGcInfoSpaceSize = 64;
  private GpuDevice mDevice;

  private long m_PreviousRef;
  private long m_PreviousSize;

  protected long mBufferSize;
  protected Memory mToSpaceMemory;
  protected Memory mTextureMemory;
  protected Memory mHandlesMemory;
  protected Memory mHeapEndPtrMemory;
  protected Memory mToHandleMapMemory;
  protected Memory mGcInfoSpaceMemory;
  protected Memory mExceptionsMemory;

  protected Serializer mGcObjectVisitor;
  private boolean mUsingGarbageCollector;
  private int m_CountWritten;
  private List<Long> m_HandlesList;

  private long mMaxToHandleMapAddress;

  private PartiallyCompletedParallelJob mWriteRet;

  private static Map<GpuDevice, GcHeap> mInstances = new HashMap<GpuDevice, GcHeap>();

  public static GcHeap v(GpuDevice device){
    if(mInstances.containsKey(device)){
      GcHeap ret = mInstances.get(device);
      ret.reset();
      return ret;
    }
    GcHeap ret = device.CreateHeap();
    //mInstances.put(device, ret);
    return ret;
  }
  
  private void reset(){
    mBlocks.clear();
    m_PreviousRef = 0;
    m_PreviousSize = 0;

    mToSpaceMemory.setAddress(0);
    mToSpaceMemory.clearHeapEndPtr();
    
    mHandlesMemory.setAddress(0);
    mHandlesMemory.clearHeapEndPtr();
    mHeapEndPtrMemory.setAddress(0);
    mHeapEndPtrMemory.clearHeapEndPtr();

    mGcObjectVisitor = null;
    mUsingGarbageCollector = false;
    m_CountWritten = 0;

    mMaxToHandleMapAddress = 0;

    mWriteRet = null;
  }
  
  protected GcHeap(GpuDevice device){
    mDevice = device;
    m_HandlesList = new ArrayList<Long>();
  }
  
  private void writeOneRuntimeBasicBlock(CompiledKernel block){
    mBlocks.add(block);
    long ref = mGcObjectVisitor.writeToHeap(block, true);
    mHandlesMemory.writeLong(ref);
    
    m_HandlesList.add(ref);
    if(mUsingGarbageCollector){
      long to_handle_map_memory_address = ref*4;
      mToHandleMapMemory.setAddress(to_handle_map_memory_address);
      if(to_handle_map_memory_address > mMaxToHandleMapAddress)
        mMaxToHandleMapAddress = to_handle_map_memory_address;
      mToHandleMapMemory.writeLong(ref);
    }
    long prev_size = ref - m_PreviousRef;
    if(prev_size > m_PreviousSize){
      m_PreviousSize = prev_size;
    }
    m_PreviousRef = ref; 
  }
  
  private CompiledKernel getBlock(Iterator<Kernel> jobs){
    Kernel job = jobs.next();
    mWriteRet.enqueueJob(job);
    return (CompiledKernel) job;
  }

  public int writeRuntimeBasicBlock(Kernel kernel_template, int num_threads){
    Stopwatch watch = new Stopwatch();
    watch.start();
    
    mBlocks = new ArrayList<CompiledKernel>();
    m_HandlesList.clear();
    
    CompiledKernel first_block = (CompiledKernel) kernel_template;
    
    //mUsingGarbageCollector = first_block.isUsingGarbageCollector();
    mUsingGarbageCollector = false;
    mGcObjectVisitor = first_block.getSerializer(mToSpaceMemory, mTextureMemory);

    mHeapEndPtrMemory.setAddress(0);
    mHandlesMemory.setAddress(0);
    mToSpaceMemory.setAddress(0);
    mToSpaceMemory.clearHeapEndPtr();
    
    if(mUsingGarbageCollector){
      makeSureReadyForUsingGarbageCollector();
    }

    //write statics
    mGcObjectVisitor.writeStaticsToHeap();
    
    m_PreviousRef = 0;
    m_PreviousSize = 0;
    m_CountWritten = 1;
    mMaxToHandleMapAddress = -1;

    writeOneRuntimeBasicBlock(first_block);
    for(int i = 1; i < num_threads; ++i){
      m_HandlesList.add(m_PreviousRef);
    }
    
    long heap_end_ptr = mToSpaceMemory.getHeapEndPtr();
    mHeapEndPtrMemory.writeLong(heap_end_ptr);
    
    mToSpaceMemory.finishCopy(mToSpaceMemory.getHeapEndPtr());    
    
    mHandlesMemory.finishCopy(m_CountWritten*8); //8 is sizeof long
    if(mUsingGarbageCollector){
      mToHandleMapMemory.finishCopy(mMaxToHandleMapAddress);
    }
    mHeapEndPtrMemory.finishCopy(8);    
       
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(mToSpaceMemory, 0, 1024);
    }
    return m_CountWritten;
  }
  
  public int writeRuntimeBasicBlocks(Iterator<Kernel> jobs){
    Stopwatch watch = new Stopwatch();
    watch.start();
    
    mBlocks = new ArrayList<CompiledKernel>();
    m_HandlesList.clear();
    
    mWriteRet = new PartiallyCompletedParallelJob(jobs);

    CompiledKernel first_block = getBlock(jobs);

    //mUsingGarbageCollector = first_block.isUsingGarbageCollector();
    mUsingGarbageCollector = false;
    mGcObjectVisitor = first_block.getSerializer(mToSpaceMemory, mTextureMemory);

    mHeapEndPtrMemory.setAddress(0);
    mHandlesMemory.setAddress(0);
    mToSpaceMemory.setAddress(0);
    mToSpaceMemory.clearHeapEndPtr();
    
    if(mUsingGarbageCollector){
      makeSureReadyForUsingGarbageCollector();
    }

    //write statics
    mGcObjectVisitor.writeStaticsToHeap();
    
    m_PreviousRef = 0;
    m_PreviousSize = 0;
    m_CountWritten = 1;
    mMaxToHandleMapAddress = -1;

    writeOneRuntimeBasicBlock(first_block);
    while(jobs.hasNext()){

      if(roomForMore(m_PreviousSize, m_PreviousRef) == false){
        break;
      }
      m_CountWritten++;

      CompiledKernel block = getBlock(jobs);
      writeOneRuntimeBasicBlock(block);
      
    }
    long heap_end_ptr = mToSpaceMemory.getHeapEndPtr();
    mHeapEndPtrMemory.writeLong(heap_end_ptr);
    
    mToSpaceMemory.finishCopy(mToSpaceMemory.getHeapEndPtr());    
    
    mHandlesMemory.finishCopy(m_CountWritten*8); //8 is sizeof long
    if(mUsingGarbageCollector){
      mToHandleMapMemory.finishCopy(mMaxToHandleMapAddress);
    }
    mHeapEndPtrMemory.finishCopy(8);    
       
    if(Configuration.getPrintMem()){
      BufferPrinter printer = new BufferPrinter();
      printer.print(mToSpaceMemory, 0, 1024);
    }
    return m_CountWritten;
  }

  protected abstract void allocateMemory();


  public void readRuntimeBasicBlock(Kernel kernel_template) {
    if(Configuration.getPrintMem()){
      BufferPrinter printer1 = new BufferPrinter();
      printer1.print(mToSpaceMemory, 0, 1024);
    }
    
    CompiledKernel first_block = (CompiledKernel) kernel_template;
    
    Stopwatch watch = new Stopwatch();
    watch.start();
    
    mHandlesMemory.setAddress(0);

    //read statics
    mToSpaceMemory.setAddress(0);    
        
    mGcObjectVisitor.readStaticsFromHeap();
    
    mExceptionsMemory.setAddress(0);
    for(int i = 0; i < m_CountWritten; ++i){
      long ref = mExceptionsMemory.readLong();
      if(ref != 0){
        long ref_num = ref >> 4;
        if(ref_num == first_block.getNullPointerNumber()){
          throw new NullPointerException(); 
        } else if(ref_num == first_block.getOutOfMemoryNumber()){
          throw new OutOfMemoryError();
        }
        mToSpaceMemory.setAddress(ref_num);
        Object except = mGcObjectVisitor.readFromHeap(null, true, ref_num);
        if(except instanceof Error){
          Error except_th = (Error) except;
          throw except_th;
        } else if(except instanceof RuntimeException){ 
          RuntimeException runtime_ex = (RuntimeException) except;
          throw runtime_ex;
        } else if(except instanceof GpuException){
          GpuException gpu_except = (GpuException) except;
          System.out.println("array: "+gpu_except.m_array);
          System.out.println("index: "+gpu_except.m_arrayIndex);
          System.out.println("length: "+gpu_except.m_arrayLength);
          System.exit(1);
        } else {
          throw new RuntimeException((Throwable) except);
        }
      }
    }
    
    long reference = m_HandlesList.get(0);
    mToSpaceMemory.setAddress(reference);
    mGcObjectVisitor.readFromHeap(kernel_template, true, reference);
        
    mHandlesMemory.finishRead();
    mToSpaceMemory.finishRead();
    if(mUsingGarbageCollector){
      mToHandleMapMemory.finishRead();
    }
  }
  
  public PartiallyCompletedParallelJob readRuntimeBasicBlocks(){    
    if(Configuration.getPrintMem()){
      BufferPrinter printer1 = new BufferPrinter();
      printer1.print(mToSpaceMemory, 0, 1024);
    }
    
    Stopwatch watch = new Stopwatch();
    watch.start();
    
    mHandlesMemory.setAddress(0);

    //read statics
    mToSpaceMemory.setAddress(0);    
        
    CompiledKernel first_block = mBlocks.get(0);
    mGcObjectVisitor.readStaticsFromHeap();
    
    mExceptionsMemory.setAddress(0);
    for(int i = 0; i < m_CountWritten; ++i){
      long ref = mExceptionsMemory.readLong();
      if(ref != 0){
        long ref_num = ref >> 4;
        if(ref_num == first_block.getNullPointerNumber()){
          throw new NullPointerException(); 
        } else if(ref_num == first_block.getOutOfMemoryNumber()){
          throw new OutOfMemoryError();
        }
        mToSpaceMemory.setAddress(ref);
        Object except = mGcObjectVisitor.readFromHeap(null, true, ref);
        if(except instanceof Error){
          Error except_th = (Error) except;
          throw except_th;
        } else if(except instanceof RuntimeException){ 
          RuntimeException runtime_ex = (RuntimeException) except;
          throw runtime_ex;
        } else if(except instanceof GpuException){
          GpuException gpu_except = (GpuException) except;
          gpu_except.throwArrayOutOfBounds();
        } else {
          throw new RuntimeException((Throwable) except);
        }
      }
    }
    
    //read instances
    for(int i = 0; i < m_CountWritten; ++i){
      CompiledKernel block = mBlocks.get(i);
      long reference = m_HandlesList.get(i);
      mToSpaceMemory.setAddress(reference);
      mGcObjectVisitor.readFromHeap(block, true, reference);
    }
        
    mHandlesMemory.finishRead();
    mToSpaceMemory.finishRead();
    if(mUsingGarbageCollector){
      mToHandleMapMemory.finishRead();
    }
        
    return mWriteRet;
  }

  protected abstract void makeSureReadyForUsingGarbageCollector();

  private boolean roomForMore(long size, long ref) {
    long next_ref = ref + size + size;
    //System.out.printf("Next_ref: "+next_ref);
    if(next_ref >= mBufferSize)
      return false;
    if((m_CountWritten * 4) + 4 >= mBufferSize)
      return false;
    if(m_CountWritten + 1 > mDevice.getNumBlocks())
      return false;
    if(mUsingGarbageCollector){
      if(mMaxToHandleMapAddress + 4 >= mBufferSize)
        return false;
    }
    return true;
  }

  int getCountWritten() {
    return m_CountWritten;
  }

  public List<CompiledKernel> getBlocks() {
    return mBlocks;
  }
}