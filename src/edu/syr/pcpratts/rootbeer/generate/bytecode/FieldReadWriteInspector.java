/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.generate.bytecode;

import edu.syr.pcpratts.rootbeer.generate.opencl.fields.OpenCLField;
import soot.SootMethod;

public class FieldReadWriteInspector {

  public FieldReadWriteInspector(SootMethod root_method){
  }

  /**
   * Returns the fields read in the bytecode
   * @param ocl_field
   * @return
   */
  public boolean fieldIsReadOnGpu(OpenCLField ocl_field){
    return true;
  }
  
  /**
   * Returns the fields written to in the bytecode
   * @param ocl_field
   * @return
   */
  public boolean fieldIsWrittenOnGpu(OpenCLField ocl_field){
    return true;
  }

}
