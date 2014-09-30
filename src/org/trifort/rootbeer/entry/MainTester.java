/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import soot.rbclassload.MethodTester;
import soot.rbclassload.RTAMethod;

public class MainTester implements MethodTester {

  public boolean matches(RTAMethod hsm) {
    if(hsm.getSignature().getSubSignatureString().equals("void main(java.lang.String[])")){
      return true;
    }
    return false;
  }
  
}
