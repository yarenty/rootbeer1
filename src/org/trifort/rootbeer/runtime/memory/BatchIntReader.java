/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime.memory;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;

import org.trifort.rootbeer.runtime.util.Stopwatch;

public class BatchIntReader {
  
  static {
    System.load(new File("native/batchint.so.1").getAbsolutePath());
  }
  
  public native void read(byte[] buffer, int length, int[] ret_buffer);
  public native void malloc(int size);
}