/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.util.CompilerRunner;
import edu.syr.pcpratts.rootbeer.util.CudaPath;
import edu.syr.pcpratts.rootbeer.util.WindowsCompile;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class ParallelCompileJob {

  private File m_generated;
  private CudaPath m_cudaPath;
  private String m_gencodeOptions;
  private boolean m_m32;
  private CompileResult m_result;
  
  public ParallelCompileJob(File generated, CudaPath cuda_path, 
    String gencode_options, boolean m32){
    
    m_generated = generated;
    m_cudaPath = cuda_path;
    m_gencodeOptions = gencode_options;
    m_m32 = m32;
  }
  
  public void compile(){
    List<byte[]> file_contents;
    try {
      String model_string = m_m32 ? "-m32" : "-m64";
      String file_string = m_m32 ? "_32" : "_64";
      File code_file = new File(RootbeerPaths.v().getRootbeerHome() + 
        "code_file" + file_string + ".ptx");
      String command;
      if (File.separator.equals("/")) {
        command = m_cudaPath.get() + "/nvcc " + model_string + " " +
          m_gencodeOptions + " -fatbin " + m_generated.getAbsolutePath() +
          " -o " + code_file.getAbsolutePath();
        CompilerRunner runner = new CompilerRunner();
        List<String> errors = runner.run(command);
        if (errors.isEmpty() == false) {
          m_result =  new CompileResult(m_m32, null, errors);
          return;
        }
      } else {
        WindowsCompile compile = new WindowsCompile();
        String nvidia_path = m_cudaPath.get();
        command = "\"" + nvidia_path + "\" " + m_gencodeOptions +
          " -fatbin \"" + m_generated.getAbsolutePath() + "\" -o \"" +
          code_file.getAbsolutePath() + "\"" + compile.endl();
        List<String> errors = compile.compile(command);
        if (errors.isEmpty() == false) {
          m_result =  new CompileResult(m_m32, null, errors);
          return;
        }
      }
      file_contents = readFile(code_file);
    } catch(Exception ex){
      file_contents = new ArrayList<byte[]>();
      ex.printStackTrace();
    }
    m_result = new CompileResult(m_m32, file_contents, new ArrayList<String>());
  }
  
  private List<byte[]> readFile(File file) throws Exception {
    InputStream is = new FileInputStream(file);
    List<byte[]> ret = new ArrayList<byte[]>();
    while(true){
      byte[] buffer = new byte[4096];
      int len = is.read(buffer);
      if(len == -1)
        break;
      byte[] short_buffer = new byte[len];
      for(int i = 0; i < len; ++i){
        short_buffer[i] = buffer[i];
      }
      ret.add(short_buffer);
    }
    return ret;
  }
  
  public CompileResult getResult(){
    return m_result;
  }
}
