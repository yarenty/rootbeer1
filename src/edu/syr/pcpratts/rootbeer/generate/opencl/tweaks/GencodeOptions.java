/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.opencl.tweaks;

//help: http://mlso.hao.ucar.edu/hao/acos/sw/cuda-sdk/shared/common.mk

import edu.syr.pcpratts.rootbeer.util.CudaPath;
import java.io.File;

public class GencodeOptions {

  public String getOptions(){
    String version = getVersion();
    String gencode_options  = "--generate-code arch=compute_11,code=\"sm_11,compute_11\" ";
               gencode_options += "--generate-code arch=compute_12,code=\"sm_12,compute_12\" ";
               gencode_options += "--generate-code arch=compute_13,code=\"sm_13,compute_13\" ";
               gencode_options += "--generate-code arch=compute_20,code=\"sm_20,compute_20\" ";
               gencode_options += "--generate-code arch=compute_20,code=\"sm_21,compute_20 \" ";
               gencode_options += "--generate-code arch=compute_30,code=\"sm_30,compute_30\" ";
               gencode_options += "--generate-code arch=compute_35,code=\"sm_35,compute_35\" ";  
    return gencode_options;
  }

  private String getVersion() {
    CudaPath cuda_path = new CudaPath();
    String nvcc_path = cuda_path.get() + "/nvcc";
    
    CmdRunner runner = new CmdRunner();
    
    return ret;
  }
  
}
