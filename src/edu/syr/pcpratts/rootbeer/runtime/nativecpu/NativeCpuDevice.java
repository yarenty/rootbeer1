/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.runtime.nativecpu;

import edu.syr.pcpratts.rootbeer.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.runtime.PartiallyCompletedParallelJob;
import edu.syr.pcpratts.rootbeer.runtime.Kernel;
import edu.syr.pcpratts.rootbeer.runtime.CompiledKernel;
import edu.syr.pcpratts.rootbeer.runtime.Serializer;
import edu.syr.pcpratts.rootbeer.runtime.gpu.GcHeap;
import edu.syr.pcpratts.rootbeer.runtime.gpu.GpuDevice;
import edu.syr.pcpratts.rootbeer.runtime.memory.BasicMemory;
import edu.syr.pcpratts.rootbeer.runtime.memory.BufferPrinter;
import edu.syr.pcpratts.rootbeer.runtime.memory.Memory;
import edu.syr.pcpratts.rootbeer.util.ResourceReader;
import java.io.File;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;
import java.util.UUID;

public class NativeCpuDevice implements GpuDevice {
  
  private List<CompiledKernel> m_Blocks;
  
  public GcHeap CreateHeap() {
    return new NativeCpuGcHeap(this);
  }

  public long getMaxEnqueueSize() {
    return 1024*1024*1024;
  }

  public void flushQueue() {
    
  }

  public PartiallyCompletedParallelJob run(Iterator<Kernel> blocks) {
    NativeCpuGcHeap heap = new NativeCpuGcHeap(this);
    int size = heap.writeRuntimeBasicBlocks(blocks);
    m_Blocks = heap.getBlocks();
    
    List<Memory> mems = heap.getMemory();    
    String lib_name = compileNativeCpuDev();
    BasicMemory to_space = (BasicMemory) mems.get(0);
    BasicMemory handles = (BasicMemory) mems.get(1);
    BasicMemory heap_end_ptr = (BasicMemory) mems.get(2);
    BasicMemory gc_info = (BasicMemory) mems.get(3);
    BasicMemory exceptions = (BasicMemory) mems.get(4);
    
    Serializer serializer = heap.getSerializer();
    runOnCpu(to_space.getBuffer(), to_space.getBuffer().size(), handles.getBuffer().get(0), heap_end_ptr.getBuffer().get(0),
      gc_info.getBuffer().get(0), exceptions.getBuffer().get(0), serializer.getClassRefArray(), size, lib_name);
    
    PartiallyCompletedParallelJob ret = heap.readRuntimeBasicBlocks();    
    return ret;
  }
  
  private native void runOnCpu(List<byte[]> to_space, int to_space_size, 
    byte[] handles, byte[] heap_end_ptr, byte[] gc_info, byte[] exceptions, 
    int[] java_lang_class_refs, int num_threads, String library_name);

  public long getMaxMemoryAllocSize() {
    return 1024*1024*1024;
  }
  
  private void extractFromNative(String filename, String nemu) throws Exception {
    String str = ResourceReader.getResource("/edu/syr/pcpratts/rootbeer/runtime2/native/"+filename);
    PrintWriter writer = new PrintWriter(nemu+filename);
    writer.println(str);
    writer.flush();
    writer.close();
  }
  
  private String compileUnix(File nemu_file, String jdk_path, String code) throws Exception {
    String nemu = nemu_file.getAbsolutePath()+File.separator;
    PrintWriter writer = new PrintWriter(nemu+"generated.c");
    writer.println(code);
    writer.flush();
    writer.close();

    String name = "libnemu";

    int status;
    String cmd;
    Process p;

    extractFromNative("NativeCpuDevice.c", nemu);
    extractFromNative("edu_syr_pcpratts_rootbeer_runtime_nativecpu_NativeCpuDevice.h", nemu);
    
    cmd = "gcc -ggdb -Wall -fPIC -g -c -I"+jdk_path+"include/ -I/usr/lib/jvm/java-6-openjdk/include/linux "+nemu+"NativeCpuDevice.c -o "+nemu+"NativeCpuDevice.o";
    p = Runtime.getRuntime().exec(cmd, null, nemu_file);
    status = p.waitFor();
    if(status != 0){
      System.out.println("Compilation failure!");
      System.out.println(cmd);
      System.exit(-1);
    }

    cmd = "gcc -ggdb -fPIC -Wall -g -c -lpthread "+nemu+"generated.c -o "+nemu+"generated.o";
    p = Runtime.getRuntime().exec(cmd, null, nemu_file);
    status = p.waitFor();
    if(status != 0){
      System.out.println("Compilation failure!");
      System.out.println(cmd);
      System.exit(-1);
    }

    cmd = "gcc -shared -Wl,-soname,"+name+" -o "+nemu+name+".so.1 "+nemu+"generated.o -lc";
    p = Runtime.getRuntime().exec(cmd, null, nemu_file);
    status = p.waitFor();
    if(status != 0){
      System.out.println("Compilation failure!");
      System.out.println(cmd);
      System.exit(-1);
    }

    cmd = "gcc -shared -Wl,-soname,nativecpudev -o "+nemu+"nativecpudev.so.1 "+nemu+"NativeCpuDevice.o "+nemu+"generated.o -lc";
    p = Runtime.getRuntime().exec(cmd, null, nemu_file);
    status = p.waitFor();
    if(status != 0){
      System.out.println("Compilation failure!");
      System.out.println(cmd);
      System.exit(-1);
    }

    File f1 = new File(nemu+"nativecpudev.so.1");
    System.load(f1.getAbsolutePath());     

    File f2 = new File(nemu+name+".so.1");
    return f2.getAbsolutePath();
  }
  
  private String compileWindows(File nemu_file, String code){
    throw new UnsupportedOperationException();
  }

  private String compileNativeCpuDev() {
    try {
      String code = m_Blocks.get(0).getCode();
      File nemu_file = new File(RootbeerPaths.v().getRootbeerHome()+"nemu");
      if(nemu_file.exists() == false){
        nemu_file.mkdirs();  
      }
      
      if ("Mac OS X".equals(System.getProperty("os.name"))){
        String jdk_path = unixSearchJdk("/System/Library/Java/JavaVirtualMachines/");
        return compileUnix(nemu_file, jdk_path, code); 
      } else if(File.separator.equals("/")){
        String jdk_path = unixSearchJdk("/usr/lib/jvm/");
        return compileUnix(nemu_file, jdk_path, code);
      } else { 
        return compileWindows(nemu_file, code);
      }      
    } catch(Exception ex){
      ex.printStackTrace();
      System.exit(0);
      return null;
    }
  }

  private String unixSearchJdk(String base_path){
    return "/usr/lib/jvm/java-6-openjdk/";
  }
  
  public long getGlobalMemSize() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  public long getNumBlocks() {
    return 1024*1024*1024;
  }
  
}
