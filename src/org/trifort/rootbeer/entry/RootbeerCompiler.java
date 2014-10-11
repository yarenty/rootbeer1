/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarInputStream;
import java.util.jar.JarOutputStream;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.configuration.RootbeerPaths;
import org.trifort.rootbeer.generate.opencl.tweaks.CudaTweaks;
import org.trifort.rootbeer.generate.opencl.tweaks.NativeCpuTweaks;
import org.trifort.rootbeer.generate.opencl.tweaks.Tweaks;
import org.trifort.rootbeer.util.*;

import pack.Pack;
import soot.*;
import soot.options.Options;
import soot.rbclassload.BytecodeFile;
import soot.rbclassload.EntryMethodTester;
import soot.rbclassload.ListClassTester;
import soot.rbclassload.ListMethodTester;
import soot.rbclassload.MethodTester;
import soot.rbclassload.RootbeerClassLoader;
import soot.util.Chain;
import soot.util.JasminOutputStream;

public class RootbeerCompiler {

  private String provider;
  private boolean enableClassRemapping;
  private EntryMethodTester entryDetector;
  private Set<String> runtimePackages;
  private String inputJarFilename;
  private String outputJarFilename;
  private String rootbeerJarFilename;
  private List<SootMethod> entryMethods;
  private List<SootMethod> kernelMethods;
  private Set<Type> newInvokes;
  
  public RootbeerCompiler(){
    clearOutputFolders();
    findRootbeerJarFilename();
    
    if(Configuration.compilerInstance().getMode() == Configuration.MODE_GPU){      
      Tweaks.setInstance(new CudaTweaks());
    } else {
      Tweaks.setInstance(new NativeCpuTweaks());
    }
    
    enableClassRemapping = true;
    runtimePackages = new HashSet<String>();
    addRuntimePackages();
  }
  
  private void findRootbeerJarFilename(){
    CurrJarName jarName = new CurrJarName();
    rootbeerJarFilename = jarName.get();
  }
  
  private void addRuntimePackages(){
    runtimePackages.add("/org/trifort/rootbeer/configuration/");
    runtimePackages.add("/org/trifort/rootbeer/entry/");
    runtimePackages.add("/org/trifort/rootbeer/generate/");
    runtimePackages.add("/org/trifort/rootbeer/runtime/");
    runtimePackages.add("/org/trifort/rootbeer/runtime2/");
    runtimePackages.add("/org/trifort/rootbeer/test/");
    runtimePackages.add("/org/trifort/rootbeer/util/");
  }
  
  public void disableClassRemapping(){
    enableClassRemapping = false; 
  }
    
  private void setupSoot(boolean runtests){
    RootbeerClassLoader.v().setUserJar(inputJarFilename);
    
    List<String> proc_dir = new ArrayList<String>();
    proc_dir.add(inputJarFilename);
    
    Options.v().set_allow_phantom_refs(true);
    Options.v().set_rbclassload(true);
    Options.v().set_prepend_classpath(true);
    Options.v().set_process_dir(proc_dir);
    if(enableClassRemapping){
      Options.v().set_rbclassload_buildcg(true);
    }
    if(rootbeerJarFilename.equals("") == false){
      Options.v().set_soot_classpath(rootbeerJarFilename);
    }
    
    RootbeerClassLoader.v().addEntryMethodTester(entryDetector);
    RootbeerClassLoader.v().addNewInvoke("java.lang.StringBuilder");
    RootbeerClassLoader.v().addSignaturesClass("java.lang.Object");
    RootbeerClassLoader.v().addSignaturesClass("java.lang.StringBuilder");
    RootbeerClassLoader.v().addSignaturesClass("org.trifort.rootbeer.runtime.Serializer");
    RootbeerClassLoader.v().addSignaturesClass("org.trifort.rootbeer.runtime.Memory");
    RootbeerClassLoader.v().addSignaturesClass("java.lang.String");
    RootbeerClassLoader.v().addSignaturesClass("org.trifort.rootbeer.runtimegpu.GpuException");
    RootbeerClassLoader.v().addSignaturesClass("org.trifort.rootbeer.runtime.Sentinal");
    RootbeerClassLoader.v().addSignaturesClass("org.trifort.rootbeer.runtime.PrivateFields");

    ListMethodTester dont_dfs_tester = new ListMethodTester();
    for(String no_dfs : CompilerSetup.getDontDfs()){
      dont_dfs_tester.addSignature(no_dfs);
    }
    RootbeerClassLoader.v().addDontFollowMethodTester(dont_dfs_tester);
    
    ForcedFields forced_fields = new ForcedFields();
    for(String field_sig : forced_fields.get()){
      RootbeerClassLoader.v().loadField(field_sig);
    }
    
    ListMethodTester toSigMethods = new ListMethodTester();
    for(String method : CompilerSetup.getToSignaturesMethods()){
      toSigMethods.addSignature(method);
    }
    RootbeerClassLoader.v().addToSignaturesMethodTester(toSigMethods);
    RootbeerClassLoader.v().addClassRemapping("java.util.concurrent.atomic.AtomicLong", "org.trifort.rootbeer.remap.GpuAtomicLong");
    RootbeerClassLoader.v().addClassRemapping("org.trifort.rootbeer.testcases.rootbeertest.remaptest.CallsPrivateMethod", "org.trifort.rootbeer.remap.DoesntCallPrivateMethod");
    RootbeerClassLoader.v().loadNecessaryClasses();
  }
  
  public void compile(String inputJarFilename, String outputJarFilename, String test_case) throws Exception {
    this.inputJarFilename = inputJarFilename;
    this.outputJarFilename = outputJarFilename;
    TestCaseEntryPointDetector detector = new TestCaseEntryPointDetector(test_case);
    entryDetector = detector;
    setupSoot(true);
    provider = detector.getProvider();
        
    entryMethods = RootbeerClassLoader.v().getEntryPoints();
    compileForKernels();
  }
  
  public void compile(String jar_filename, String outname) throws Exception {
    compile(jar_filename, outname, false);
  }
  
  public void compile(String jar_filename, String outname, boolean run_tests) throws Exception {
    entryDetector = new KernelEntryPointDetector(run_tests);
    setupSoot(run_tests);
    
    entryMethods = RootbeerClassLoader.v().getEntryPoints();
    compileForKernels();
  }
  
  private void compileForKernels() throws Exception {
    findKernelMethods();
    
    if(kernelMethods.isEmpty()){
      System.out.println("There are no kernel classes. Please implement the following interface to use rootbeer:");
      System.out.println("org.trifort.rootbeer.runtime.Kernel");
      System.exit(0);
    }
       
    KernelTransform kernelTransform = new KernelTransform();
    for(SootMethod kernelMethod : kernelMethods){   
      DfsInfo.reset();
      
      System.out.println("running KernelTransform on: "+kernelMethod.getSignature()+"...");
      DfsInfo.v().setVirtualMethodBases(newInvokes);
      RootbeerDfs rootbeerDfs = new RootbeerDfs();
      rootbeerDfs.run(kernelMethod.getSignature());
      DfsInfo.v().expandArrayTypes();
      DfsInfo.v().finalizeTypes();

      SootClass sootClass = kernelMethod.getDeclaringClass();
      kernelTransform.run(sootClass.getName());
    }
    
    System.out.println("writing classes out...");
    emitOutput();
  }
  
  private void findKernelMethods(){
    kernelMethods = new ArrayList<SootMethod>();
    newInvokes = new TreeSet<Type>();
    for(SootMethod entryMethod : entryMethods){
      DfsInfo.reset();
      DfsInfo.v().setVirtualMethodBases(newInvokes);
      RootbeerDfs rootbeerDfs = new RootbeerDfs();
      rootbeerDfs.run(entryMethod.getSignature());
      newInvokes.addAll(DfsInfo.v().getNewInvokes());
      
      for(SootMethod sootMethod : DfsInfo.v().getMethods()){
        SootClass declaringClass = sootMethod.getDeclaringClass();
        Chain<SootClass> interfaces = declaringClass.getInterfaces();
        for(SootClass iface : interfaces){
          if(iface.getName().equals("org.trifort.rootbeer.runtime.Kernel")){
            if(sootMethod.getSubSignature().equals("void gpuMethod()")){
              if(kernelMethods.contains(sootMethod) == false){
                System.out.println("adding kernel method: "+sootMethod.getSignature());
                kernelMethods.add(sootMethod);
              }
            }
          }
        }
      }
    }
  }
  
  private void emitOutput() throws Exception {
    Map<String, byte[]> rootbeerFiles = readJar(rootbeerJarFilename);
    Map<String, byte[]> applicationFiles = readJar(inputJarFilename);
    for(String filename : rootbeerFiles.keySet()){
      if(isRuntimePackage(filename)){
        applicationFiles.put(filename, rootbeerFiles.get(filename));
      }
    }
    List<BytecodeFile> bytecodeFiles = RootbeerClassLoader.v().getModifiedBytecodeFiles();
    for(BytecodeFile bytecodeFile : bytecodeFiles){
      applicationFiles.put(bytecodeFile.getFileName(), bytecodeFile.getContents());
    }
    applicationFiles.put("org/trifort/rootbeer/runtime/config.txt", getConfigurationFile());
    Map<String, byte[]> cubinFiles = CubinFiles.v().getCubinFiles();
    System.out.println("CUBIN_FILES_SIZE: "+cubinFiles.size());
    for(String cubinFilename : cubinFiles.keySet()){
      byte[] cubinFile = cubinFiles.get(cubinFilename);
      System.out.println("cubinFilename: "+cubinFilename+" length: "+cubinFile.length);
      applicationFiles.put(cubinFilename, cubinFiles.get(cubinFilename));
    }
    writeJar(applicationFiles, outputJarFilename);
  }
  
  private boolean isRuntimePackage(String filename){
    for(String runtimePackage : runtimePackages){
      if(filename.startsWith(runtimePackage)){
        return true;
      }
    }
    return false;
  }
  
  private Map<String, byte[]> readJar(String inputFilename) throws Exception {
    Map<String, byte[]> ret = new TreeMap<String, byte[]>();
    JarInputStream jarInput = new JarInputStream(new FileInputStream(inputFilename));
    while(true){
      JarEntry jarEntry = jarInput.getNextJarEntry();
      if(jarEntry == null){
        break;
      }
      String filename = jarEntry.getName();
      byte[] contents = RootbeerClassLoader.v().readFully(jarInput);
      ret.put(filename, contents);
    }
    jarInput.close();
    return ret;
  }
  
  private void writeJar(Map<String, byte[]> entries, String outputFilename) throws Exception {
    JarOutputStream jarOutput = new JarOutputStream(new FileOutputStream(outputFilename));
    for(String filename : entries.keySet()){
      String name = filename;
      byte[] contents = entries.get(filename);
      JarEntry jarEntry = new JarEntry(name);
      jarEntry.setSize(contents.length);
      jarEntry.setCrc(calcCRC32(contents));
      jarOutput.putNextEntry(jarEntry);
      
      int wroteLen = 0;
      int totalLen = contents.length;
      while(wroteLen < totalLen){
        int bufferSize = 4096;
        int lengthLeft = totalLen - wroteLen;
        if(bufferSize > lengthLeft){
          bufferSize = lengthLeft;
        }
        jarOutput.write(contents, wroteLen, bufferSize);
        wroteLen += bufferSize;
      }
      jarOutput.flush();
    }
    jarOutput.close();
  }
  
  private byte[] getConfigurationFile(){
    byte[] contents = new byte[2];
    contents[0] = (byte) Configuration.compilerInstance().getMode();
    if(Configuration.compilerInstance().getExceptions()){
      contents[1] = (byte) 1;
    } else {
      contents[1] = (byte) 0;
    }
    return contents;
  }

  private long calcCRC32(byte[] buffer){
    CRC32 crc = new CRC32();
    crc.update(buffer);
    return crc.getValue();
  }
  
  private void clearOutputFolders() {
    DeleteFolder deleter = new DeleteFolder();
    deleter.delete(RootbeerPaths.v().getOutputJarFolder());
    deleter.delete(RootbeerPaths.v().getOutputClassFolder());
    deleter.delete(RootbeerPaths.v().getOutputShimpleFolder());
    deleter.delete(RootbeerPaths.v().getJarContentsFolder());
  }

  public String getProvider() {
    return provider;
  }
}
