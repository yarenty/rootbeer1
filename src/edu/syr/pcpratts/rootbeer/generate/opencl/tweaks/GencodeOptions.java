/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

//help: http://mlso.hao.ucar.edu/hao/acos/sw/cuda-sdk/shared/common.mk

import edu.syr.pcpratts.rootbeer.util.CmdRunner;
import edu.syr.pcpratts.rootbeer.util.CudaPath;
import java.io.File;
import java.util.List;

public class GencodeOptions {

  public String getOptions(){
    String version = getVersion();
    String sm_35 = "--generate-code arch=compute_35,code=\"sm_35,compute_35\" ";
    String sm_30 = "--generate-code arch=compute_30,code=\"sm_30,compute_30\" ";
    String sm_21 = "--generate-code arch=compute_20,code=\"sm_21,compute_20 \" ";
    String sm_20 = "--generate-code arch=compute_20,code=\"sm_20,compute_20\" ";
    String sm_13 = "--generate-code arch=compute_13,code=\"sm_13,compute_13\" ";
    String sm_12 = "--generate-code arch=compute_12,code=\"sm_12,compute_12\" ";
    String sm_11 = "--generate-code arch=compute_11,code=\"sm_11,compute_11\" ";
    
    if(version.equals("Cuda compilation tools, release 5.0, V0.2.1221")){
      return sm_35 = sm_30 + sm_21 + sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 4.2, V0.2.1221")){
      return sm_30 + sm_21 + sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 4.1, V0.2.1221")){
      return sm_30 + sm_21 + sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 4.0, V0.2.1221")){
      return sm_30 + sm_21 + sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 3.2, V0.2.1221")){
      return sm_30 + sm_21 + sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 3.1, V0.2.1221")){
      return sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 3.0, V0.2.1221")){
      return sm_20 + sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 2.3, V0.2.1221")){
      return sm_13 + sm_12 + sm_11;
    } else if(version.equals("Cuda compilation tools, release 2.2, V0.2.1221")){
      return sm_13 + sm_12 + sm_11;
    } else {
      throw new RuntimeException("unsupported nvcc version. please send an email to pcpratts@trifort.org");
    }
  }

  private String getVersion() {
    CudaPath cuda_path = new CudaPath();
    String nvcc_path = cuda_path.get() + "/nvcc";
    
    CmdRunner runner = new CmdRunner();
    runner.run(nvcc_path+" --version", new File("."));
    
    List<String> lines = runner.getOutput();
    if(lines.isEmpty()){
      throw new RuntimeException("error detecting nvcc version.");
    }
    
    String last_line = lines.get(lines.size()-1);
    return last_line;
  }
  
}
