/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

import edu.syr.pcpratts.rootbeer.util.CudaPath;
import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class GenerateClScript {

  private List<String> m_VisualStudioPaths;
  
  public GenerateClScript(){
    m_VisualStudioPaths = new ArrayList<String>();
    m_VisualStudioPaths.add("D:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\vcvarsall.bat");
    m_VisualStudioPaths.add("D:\\Program Files\\Microsoft Visual Studio 10.0\\VC\\vcvarsall.bat");
    m_VisualStudioPaths.add("C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC\\vcvarsall.bat");
    m_VisualStudioPaths.add("C:\\Program Files\\Microsoft Visual Studio 10.0\\VC\\vcvarsall.bat");
  }
  
  public File execute(File generated, File code_file) {
    String vs_path = findPath(m_VisualStudioPaths, "Visual Studio");
    CudaPath cuda_path = new CudaPath();
    String nvidia_path = cuda_path.get();
        
    String file_text = "";
    String endl = System.getProperty("line.separator"); 
    file_text += "@call \""+vs_path+"\" amd64"+endl;
    file_text += "\""+nvidia_path+"\" -arch sm_20 -cubin \""+generated.getAbsolutePath()+"\" -o \""+code_file.getAbsolutePath()+"\""+endl;
    File ret = new File("cl_script.bat");
    try {
      PrintWriter writer = new PrintWriter(ret);
      writer.println(file_text);
      writer.flush();
      writer.close();
      return ret;
    } catch(Exception ex){
      throw new RuntimeException(ex);
    }
  }

  private String findPath(List<String> paths, String desc) {
    for(String path : paths){
      File file = new File(path);
      if(file.exists())
        return path;
    }
    throw new RuntimeException("cannot find path for: "+desc);
  }
  
}
