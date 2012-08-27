/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler;

import edu.syr.pcpratts.rootbeer.Constants;
import edu.syr.pcpratts.rootbeer.compiler.ClassRemapping;
import edu.syr.pcpratts.rootbeer.util.DeleteFolder;
import edu.syr.pcpratts.rootbeer.util.JarToFolder;
import edu.syr.pcpratts.rootbeer.util.ReadJar;
import edu.syr.pcpratts.rootbeer.util.StringDelegate;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class RootbeerCompilerSetup implements StringDelegate {
  private List<String> m_AllClasses;
  
  public List<String> setup(String single_jar_file){
    List<String> jar_files = new ArrayList<String>();
    jar_files.add(single_jar_file);
    return setup(jar_files);
  }
  
  public List<String> setup(List<String> jar_files) {
    m_AllClasses = new ArrayList<String>();
    clearOutputFolders();
    writeJars(jar_files);
    writeRuntimeClasses();
    return m_AllClasses;
  }  
  
  private void writeJars(List<String> jar_files) {
    String folder = Constants.JAR_CONTENTS_FOLDER;
    File folder_file = new File(folder);
    folder_file.mkdirs();

    try {
      JarToFolder jtf = new JarToFolder(this);
      for(String jar : jar_files){
        jtf.writeJar(jar, folder);
      } 
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }

  private void writeRuntimeClasses() {
    List<String> classes = getRuntimeClasses();
    ClassRemapping remapping = new ClassRemapping();
    classes.addAll(remapping.getRuntimeClassesJar());
    for(String cls : classes){
      writeRuntimeClass(cls);
    }
  }
  
  private List<String> getRuntimeClasses(){
    List<String> ret = new ArrayList<String>();
    ret.add("/edu/syr/pcpratts/rootbeer/generate/bytecode/Constants.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/RootbeerFactory.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/Rootbeer.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/RootbeerGpu.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/Kernel.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/CompiledKernel.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/Serializer.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/memory/Memory.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/Sentinal.class");
    ret.add("/edu/syr/pcpratts/rootbeer/test/TestSerialization.class");
    ret.add("/edu/syr/pcpratts/rootbeer/test/TestSerializationFactory.class");
    ret.add("/edu/syr/pcpratts/rootbeer/test/TestException.class");
    ret.add("/edu/syr/pcpratts/rootbeer/test/TestExceptionFactory.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/util/Stopwatch.class");
    ret.add("/edu/syr/pcpratts/rootbeer/runtime/PrivateFields.class");
    m_AllClasses.addAll(ret);
    return ret;
  }

  private void writeRuntimeClass(String cls) {
    ReadJar reader = new ReadJar();
    reader.writeRuntimeClass(cls);
  }
  
  private void clearOutputFolders() {
    DeleteFolder deleter = new DeleteFolder();
    deleter.delete(Constants.OUTPUT_JAR_FOLDER);
    deleter.delete(Constants.OUTPUT_CLASS_FOLDER);
    deleter.delete(Constants.OUTPUT_SHIMPLE_FOLDER);
  }

  public void call(String value) {
    m_AllClasses.add(value);
  }
}
