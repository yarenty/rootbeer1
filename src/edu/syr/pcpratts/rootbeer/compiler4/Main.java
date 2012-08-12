/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.compiler4;

public class Main {

  public static void main(String[] args){
    RootbeerCompiler4 compiler = new RootbeerCompiler4();
    compiler.compile("Rootbeer.jar", "output.jar");
  }
}
