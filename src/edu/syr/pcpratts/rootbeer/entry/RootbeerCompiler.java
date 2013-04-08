/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.entry;

import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.configuration.Configuration;
import edu.syr.pcpratts.rootbeer.compiler.*;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.CudaTweaks;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.NativeCpuTweaks;
import edu.syr.pcpratts.rootbeer.generate.opencl.tweaks.Tweaks;
import edu.syr.pcpratts.rootbeer.util.*;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.*;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import pack.Pack;
import soot.*;
import soot.options.Options;
import soot.rbclassload.EntryPointDetector;
import soot.rbclassload.RootbeerClassLoader;
import soot.util.JasminOutputStream;

public class RootbeerCompiler {

  private String m_classOutputFolder;
  private String m_jimpleOutputFolder;
  private String m_provider;
  private boolean m_enableClassRemapping;
  private EntryPointDetector m_entryDetector;
  private Set<String> m_runtimeClasses;
  
  public RootbeerCompiler(){
    clearOutputFolders();
    
    m_classOutputFolder = RootbeerPaths.v().getOutputClassFolder();
    m_jimpleOutputFolder = RootbeerPaths.v().getOutputJimpleFolder();
    
    if(Configuration.compilerInstance().getMode() == Configuration.MODE_GPU){      
      Tweaks.setInstance(new CudaTweaks());
    } else {
      Tweaks.setInstance(new NativeCpuTweaks());
    }
    
    m_enableClassRemapping = true;
    m_runtimeClasses = new HashSet<String>();
    addRuntimeClasses();
  }
  
  private void addRuntimeClasses(){
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtimegpu.GpuException");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.generate.bytecode.Constants");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.RootbeerFactory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.Rootbeer");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.StatsRow");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.RootbeerGpu");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.Kernel");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.CompiledKernel");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.Serializer");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.memory.Memory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.Sentinal");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.ThreadConfig");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.RootbeerTestAgent");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestSerialization");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestSerializationFactory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestException");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestExceptionFactory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestKernelTemplate");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestKernelTemplateFactory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestApplication");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.test.TestApplicationFactory");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.util.Stopwatch");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.runtime.PrivateFields");
    m_runtimeClasses.add("edu.syr.pcpratts.rootbeer.util.IntStack");
  }
  
  public void disableClassRemapping(){
    m_enableClassRemapping = false; 
  }
  
  public void compile(String main_jar, List<String> lib_jars, List<String> dirs, String dest_jar) {
    
  }
    
  private void setupSoot(String jar_filename, String rootbeer_jar, boolean runtests){
    RootbeerClassLoader.v().setUserJar(jar_filename);
    extractJar(jar_filename);
    
    List<String> proc_dir = new ArrayList<String>();
    proc_dir.add(RootbeerPaths.v().getJarContentsFolder());
    
    Options.v().set_allow_phantom_refs(true);
    Options.v().set_rbclassload(true);
    Options.v().set_prepend_classpath(true);
    Options.v().set_process_dir(proc_dir);
    if(m_enableClassRemapping){
      Options.v().set_rbclassload_buildcg(true);
    }
    if(rootbeer_jar.equals("") == false){
      Options.v().set_soot_classpath(rootbeer_jar);
    }
    
    //Options.v().set_rbcl_remap_all(Configuration.compilerInstance().getRemapAll());
    Options.v().set_rbcl_remap_all(false);
    Options.v().set_rbcl_remap_prefix("edu.syr.pcpratts.rootbeer.runtime.remap.");
    
    RootbeerClassLoader.v().addEntryPointDetector(m_entryDetector);
    
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.compressor.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.deadmethods.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.jpp.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.compiler.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.configuration.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.entry.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.generate.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.test.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.testcases.");
    RootbeerClassLoader.v().addIgnorePackage("edu.syr.pcpratts.rootbeer.util.");
    RootbeerClassLoader.v().addIgnorePackage("pack.");
    RootbeerClassLoader.v().addIgnorePackage("jasmin.");
    RootbeerClassLoader.v().addIgnorePackage("soot.");
    RootbeerClassLoader.v().addIgnorePackage("beaver.");
    RootbeerClassLoader.v().addIgnorePackage("polyglot.");
    RootbeerClassLoader.v().addIgnorePackage("org.antlr.");
    RootbeerClassLoader.v().addIgnorePackage("java_cup.");
    RootbeerClassLoader.v().addIgnorePackage("ppg.");
    RootbeerClassLoader.v().addIgnorePackage("antlr.");
    RootbeerClassLoader.v().addIgnorePackage("jas.");
    RootbeerClassLoader.v().addIgnorePackage("scm.");
    RootbeerClassLoader.v().addIgnorePackage("org.xmlpull.v1.");
    RootbeerClassLoader.v().addIgnorePackage("android.util.");
    RootbeerClassLoader.v().addIgnorePackage("android.content.res.");
    RootbeerClassLoader.v().addIgnorePackage("org.apache.commons.codec.");
    
    if(runtests){
      RootbeerClassLoader.v().addTestCasePackage("edu.syr.pcpratts.rootbeer.testcases.");
    }
    
    RootbeerClassLoader.v().addKeepPackages("edu.syr.pcpratts.rootbeer.runtime2.");
    RootbeerClassLoader.v().addKeepPackages("edu.syr.pcpratts.rootbeer.configuration.");
    
    for(String runtime_class : m_runtimeClasses){
      RootbeerClassLoader.v().addSignaturesClass(runtime_class);
    }
    
    RootbeerClassLoader.v().addNewInvoke("java.lang.StringBuilder");
    RootbeerClassLoader.v().addFollowSignature("<java.lang.StringBuilder: void <init>()>");
    RootbeerClassLoader.v().addFollowSignature("<java.lang.StringBuilder: java.lang.StringBuilder append(java.lang.String)>");
    RootbeerClassLoader.v().addFollowSignature("java.lang.StringBuilder: java.lang.String toString()>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.Sentinal: void <init>()>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtimegpu.GpuException: edu.syr.pcpratts.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.Serializer: void <init>(edu.syr.pcpratts.rootbeer.runtime.memory.Memory,edu.syr.pcpratts.rootbeer.runtime.memory.Memory)>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.memory.Memory: void useInstancePointer()>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.memory.Memory: void useStaticPointer()>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.memory.Memory: long mallocWithSize(int)>");
    RootbeerClassLoader.v().addFollowSignature("<edu.syr.pcpratts.rootbeer.runtime.memory.Memory: void setAddress(long)>");
    
    RootbeerClassLoader.v().addConditionalCudaEntry(new StringConstantCudaEntry());
    
    DontDfsMethods dont_dfs_methods = new DontDfsMethods();
    Set<String> dont_dfs_set = dont_dfs_methods.get();
    for(String dont_dfs : dont_dfs_set){
      RootbeerClassLoader.v().addDontDfsMethod(dont_dfs);
    }
    
    List<String> cuda_fields = new ArrayList<String>();
    cuda_fields.add("<java.lang.Class: java.lang.String name>");
    RootbeerClassLoader.v().setCudaFields(cuda_fields);
    
    RootbeerClassLoader.v().loadNecessaryClasses();
  }
  
  public void compile(String jar_filename, String outname, String test_case) throws Exception {
    TestCaseEntryPointDetector detector = new TestCaseEntryPointDetector(test_case);
    m_entryDetector = detector;
    CurrJarName jar_name = new CurrJarName();
    setupSoot(jar_filename, jar_name.get(), true);
    m_provider = detector.getProvider();
        
    List<SootMethod> kernel_methods = RootbeerClassLoader.v().getEntryPoints();
    compileForKernels(outname, kernel_methods);
  }
  
  public void compile(String jar_filename, String outname) throws Exception {
    compile(jar_filename, outname, false);
  }
  
  public void compile(String jar_filename, String outname, boolean run_tests) throws Exception {
    m_entryDetector = new KernelEntryPointDetector();
    CurrJarName jar_name = new CurrJarName();
    setupSoot(jar_filename, jar_name.get(), run_tests);
    
    List<SootMethod> kernel_methods = RootbeerClassLoader.v().getEntryPoints();
    compileForKernels(outname, kernel_methods);
  }
  
  private void compileForKernels(String outname, List<SootMethod> kernel_methods) throws Exception {
    
    if(kernel_methods.isEmpty()){
      System.out.println("There are no kernel classes. Please implement the following interface to use rootbeer:");
      System.out.println("edu.syr.pcpratts.rootbeer.runtime.Kernel");
      System.exit(0);
    }
    
    System.out.println("applying optimizations...");
    RootbeerClassLoader.v().applyOptimizations();
   
    Transform2 transform2 = new Transform2();
    for(SootMethod kernel_method : kernel_methods){   
      System.out.println("running transform2 on: "+kernel_method.getSignature()+"...");
      RootbeerClassLoader.v().loadDfsInfo(kernel_method);
      SootClass soot_class = kernel_method.getDeclaringClass();
      transform2.run(soot_class.getName());
    }
    
    System.out.println("writing classes out...");
    
    Iterator<SootClass> iter = Scene.v().getClasses().iterator();
    while(iter.hasNext()){
      SootClass soot_class = iter.next();
      if(soot_class.isLibraryClass()){
        System.out.println("skipping library class: "+soot_class.getName());
        continue;
      }
      String class_name = soot_class.getName();
      if(m_runtimeClasses.contains(class_name)){
        System.out.println("skipping runtine class: "+class_name);
        continue;
      }
      System.out.println("writing class_file: "+class_name);
      writeClassFile(class_name);
      writeJimpleFile(class_name);
    }
    
    makeOutJar();
    pack(outname);
  }
  
  public void pack(String outjar_name) throws Exception {
    Pack p = new Pack();
    String main_jar = RootbeerPaths.v().getOutputJarFolder() + File.separator + "partial-ret.jar";
    List<String> lib_jars = new ArrayList<String>();
    CurrJarName jar_name = new CurrJarName();
    lib_jars.add(jar_name.get());
    p.run(main_jar, lib_jars, outjar_name);
  }

  public void makeOutJar() throws Exception {
    JarEntryHelp.mkdir(RootbeerPaths.v().getOutputJarFolder() + File.separator);
    String outfile = RootbeerPaths.v().getOutputJarFolder() + File.separator + "partial-ret.jar";

    ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(outfile));
    addJarInputManifestFiles(zos);
    addOutputClassFiles(zos);
    addConfigurationFile(zos);
    zos.flush();
    zos.close();
  }
  
  private void addJarInputManifestFiles(ZipOutputStream zos) throws Exception {
    List<File> jar_input_files = getFiles(RootbeerPaths.v().getJarContentsFolder());
    for(File f : jar_input_files){
      if(f.getPath().contains("META-INF")){
        writeFileToOutput(f, zos, RootbeerPaths.v().getJarContentsFolder());
      }
    }
  }

  private void addOutputClassFiles(ZipOutputStream zos) throws Exception {
    List<File> output_class_files = getFiles(RootbeerPaths.v().getOutputClassFolder());
    for(File f : output_class_files){
      writeFileToOutput(f, zos, RootbeerPaths.v().getOutputClassFolder());
    }
  }
  
  private List<File> getFiles(String path) {
    File f = new File(path);
    List<File> ret = new ArrayList<File>();
    getFiles(ret, f);
    return ret;
  }
  
  private void getFiles(List<File> total_files, File dir){
    File[] files = dir.listFiles();
    for(File f : files){
      if(f.isDirectory()){
        getFiles(total_files, f);
      } else {
        total_files.add(f);
      }
    }
  }

  private String makeJarFileName(File f, String folder) {
    try {
      String abs_path = f.getAbsolutePath();
      if(f.isDirectory()){
        abs_path += File.separator; 
      }
      folder += File.separator;
      folder = folder.replace("\\", "\\\\");
      String[] tokens = abs_path.split(folder);
      String ret = tokens[1];
      if(File.separator.equals("\\")){
        ret = ret.replace("\\", "/");
      }
      return ret;
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }

  private void addConfigurationFile(ZipOutputStream zos) throws IOException {
    String folder_name = "edu/syr/pcpratts/rootbeer/runtime/";
    String name = folder_name + "config.txt";
    ZipEntry entry = new ZipEntry(name);
    entry.setSize(1);
    byte[] contents = new byte[1];
    contents[0] = (byte) Configuration.compilerInstance().getMode();
    
    entry.setCrc(calcCrc32(contents));
    zos.putNextEntry(entry);
    zos.write(contents);
    zos.flush();
    
    File file = new File(RootbeerPaths.v().getOutputClassFolder()+File.separator+folder_name);
    if(file.exists() == false){
      file.mkdirs();
    }
    
    FileOutputStream fout = new FileOutputStream(RootbeerPaths.v().getOutputClassFolder()+File.separator+name);
    fout.write(contents);
    fout.flush();
    fout.close();
  }
  
  private void writeFileToOutput(File f, ZipOutputStream zos, String folder) throws Exception {
    String name = makeJarFileName(f, folder);
    ZipEntry entry = new ZipEntry(name);
    byte[] contents = readFile(f);
    entry.setSize(contents.length);

    entry.setCrc(calcCrc32(contents));
    zos.putNextEntry(entry);

    int wrote_len = 0;
    int total_len = contents.length;
    while(wrote_len < total_len){
      int len = 4096;
      int len_left = total_len - wrote_len;
      if(len > len_left)
        len = len_left;
      zos.write(contents, wrote_len, len);
      wrote_len += len;
    }
    zos.flush();
  }

  private long calcCrc32(byte[] buffer){
    CRC32 crc = new CRC32();
    crc.update(buffer);
    return crc.getValue();
  }

  private byte[] readFile(File f) throws Exception {
    List<Byte> contents = new ArrayList<Byte>();
    byte[] buffer = new byte[4096];
    FileInputStream fin = new FileInputStream(f);
    while(true){
      int len = fin.read(buffer);
      if(len == -1)
        break;
      for(int i = 0; i < len; ++i){
        contents.add(buffer[i]);
      }
    }
    fin.close();
    byte[] ret = new byte[contents.size()];
    for(int i = 0; i < contents.size(); ++i)
      ret[i] = contents.get(i);
    return ret;
  }

  private void writeJimpleFile(String cls){  
    try {
      SootClass c = Scene.v().getSootClass(cls);
      JimpleWriter writer = new JimpleWriter();
      writer.write(classNameToFileName(cls, true), c);
    } catch(Exception ex){
      System.out.println("Error writing .jimple: "+cls);
    }   
  }
  
  private void writeClassFile(String cls, String filename){
    if(cls.equals("java.lang.Object"))
      return;
    FileOutputStream fos = null;
    PrintWriter writer = null;
    SootClass c = Scene.v().getSootClass(cls);
    List<String> before_sigs = getMethodSignatures(c);
    try {
      fos = new FileOutputStream(filename);
      OutputStream out1 = new JasminOutputStream(fos);
      writer = new PrintWriter(new OutputStreamWriter(out1));
      new soot.jimple.JasminClass(c).print(writer);
    } catch(Exception ex){
      System.out.println("Error writing .class: "+cls);
      ex.printStackTrace(System.out);
      List<String> after_sigs = getMethodSignatures(c);
      System.out.println("Before sigs: ");
      printMethodSigs(before_sigs);
      System.out.println("After sigs: ");
      printMethodSigs(after_sigs);
    } finally { 
      writer.flush();
      writer.close();
      try {
        fos.close(); 
      } catch(Exception ex){ }
    }
  }
  
  private List<String> getMethodSignatures(SootClass c){
    List<String> ret = new ArrayList<String>();
    List<SootMethod> methods = c.getMethods();
    for(SootMethod method : methods){
      ret.add(method.getSignature());
    }
    return ret;
  }
  
  private void printMethodSigs(List<String> sigs){
    for(String sig : sigs){
      System.out.println("  "+sig);
    }
  }
  
  private void writeClassFile(String cls) {
    writeClassFile(cls, classNameToFileName(cls, false));
  }
  
  private String classNameToFileName(String cls, boolean jimple){
    File f;
    if(jimple)
      f = new File(m_jimpleOutputFolder);
    else
      f = new File(m_classOutputFolder);
    
    cls = cls.replace(".", File.separator);
    
    if(jimple)
      cls += ".jimple";
    else
      cls += ".class";
    
    cls = f.getAbsolutePath()+File.separator + cls;
    
    File f2 = new File(cls);
    String folder = f2.getParent();
    new File(folder).mkdirs();
    
    return cls;
  }
  
  private void clearOutputFolders() {
    DeleteFolder deleter = new DeleteFolder();
    deleter.delete(RootbeerPaths.v().getOutputJarFolder());
    deleter.delete(RootbeerPaths.v().getOutputClassFolder());
    deleter.delete(RootbeerPaths.v().getOutputShimpleFolder());
    deleter.delete(RootbeerPaths.v().getJarContentsFolder());
  }

  public String getProvider() {
    return m_provider;
  }

  private void extractJar(String jar_filename) {
    JarToFolder extractor = new JarToFolder();
    try {
      System.out.println("extracting jar "+jar_filename+"...");
      extractor.writeJar(jar_filename, RootbeerPaths.v().getJarContentsFolder());
    } catch(Exception ex){
      ex.printStackTrace();
      System.exit(0);
    }
  }
}
