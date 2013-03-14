/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import edu.syr.pcpratts.rootbeer.util.WindowsCompile;
import edu.syr.pcpratts.compressor.Compressor;
import edu.syr.pcpratts.deadmethods.DeadMethods;
import edu.syr.pcpratts.rootbeer.configuration.RootbeerPaths;
import edu.syr.pcpratts.rootbeer.util.CompilerRunner;
import edu.syr.pcpratts.rootbeer.util.CudaPath;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class CudaTweaks extends Tweaks {

  @Override
  public String getGlobalAddressSpaceQualifier() {
    return "";
  }

  @Override
  public String getUnixHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaHeader.c";
  }
  
  @Override
  public String getWindowsHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaHeader.c";
  }
  
  @Override
  public String getBothHeaderPath() {
    return null;
  }
  
  @Override
  public String getGarbageCollectorPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/GarbageCollector.c";
  }

  @Override
  public String getUnixKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaKernel.c";
  }
  
  @Override
  public String getWindowsKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/CudaKernel.c";
  }

  @Override
  public String getBothKernelPath() {
    return null;
  }

  private CompileResult compileProgram0(File generated,
          CudaPath cuda_path, String gencode_options, boolean m32) throws Exception {
    File code_file = new File(RootbeerPaths.v().getRootbeerHome() + "code_file.ptx");
    String modelString = m32 ? "-m32" : "-m64";
    String command;
    if (File.separator.equals("/")) {
      command = cuda_path.get() + "/nvcc " + modelString + " " +
              gencode_options + " -fatbin " + generated.getAbsolutePath() +
              " -o " + code_file.getAbsolutePath();
      CompilerRunner runner = new CompilerRunner();
      List<String> errors = runner.run(command);
      if (errors.isEmpty() == false) {
        return new CompileResult(m32, null, errors);
      }
    } else {
      WindowsCompile compile = new WindowsCompile();
      String nvidia_path = cuda_path.get();
      command = "\"" + nvidia_path + "\" " + gencode_options +
              " -fatbin \"" + generated.getAbsolutePath() + "\" -o \"" +
              code_file.getAbsolutePath() + "\"" + compile.endl();
      List<String> errors = compile.compile(command);
      if (errors.isEmpty() == false) {
        return new CompileResult(m32, null, errors);
      }
    }
    List<byte[]> file_contents;
    try {
      file_contents = readFile(code_file);
    } catch (FileNotFoundException ex) {
      file_contents = new ArrayList<byte[]>();
      ex.printStackTrace();
    }
    return new CompileResult(m32, file_contents, new ArrayList<String>());
  }

  /**
   * Compiles CUDA code.
   *
   * @param cuda_code string containing code.
   * @return an array containing compilation results, first element of the array
   * contains 32 bit code, second element of the array contains 64 bit code. If
   * compilation for an architecture fails, only the offending element is returned.
   */
  public CompileResult[] compileProgram(String cuda_code) {
    PrintWriter writer;
    try {
      writer = new PrintWriter(RootbeerPaths.v().getRootbeerHome() + "pre_dead.cu");
      writer.println(cuda_code);
      writer.flush();
      writer.close();

      DeadMethods dead_methods = new DeadMethods("entry");
      cuda_code = dead_methods.filter(cuda_code);

      //Compressor compressor = new Compressor();
      //cuda_code = compressor.compress(cuda_code);

      File generated = new File(RootbeerPaths.v().getRootbeerHome() + "generated.cu");
      writer = new PrintWriter(generated);
      writer.println(cuda_code.toString());
      writer.flush();
      writer.close();

      CudaPath cuda_path = new CudaPath();
      GencodeOptions options_gen = new GencodeOptions();
      String gencode_options = options_gen.getOptions();
      CompileResult res32 = compileProgram0(generated, cuda_path, gencode_options, true);
      CompileResult res64 = compileProgram0(generated, cuda_path, gencode_options, false);
      return new CompileResult[] { res32, res64 };
    } catch (Exception ex) {
      throw new RuntimeException("Failed to compile cuda code.", ex);
    }
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

  @Override
  public String getDeviceFunctionQualifier() {
    return "__device__";
  }

}
