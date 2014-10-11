/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.configuration.RootbeerPaths;

public class ResourceReader {

  public static String getResource(String path) throws IOException {
    InputStream is = ResourceReader.class.getResourceAsStream(path);
    StringBuilder ret = new StringBuilder();
    BufferedReader reader = new BufferedReader(new InputStreamReader(is));
    while(true){
      String line = reader.readLine();
      if(line == null)
        break;
      ret.append(line + "\n");
    }
    is.close();
    return ret.toString();
  }


  public static byte[] getResourceArray(String jar_path, int length) throws IOException {
	  jar_path = jar_path.replace("\\", "/");
	  if(jar_path.startsWith("/") == false){
	    jar_path = "/" + jar_path;
	  }
    InputStream is = ResourceReader.class.getResourceAsStream(jar_path);
    byte[] ret = new byte[length];
    int offset = 0;
    while(offset < length){
      int thisLength = length - offset;
      int readLength = is.read(ret, offset, thisLength);
      if(readLength == -1){
        break;
      }
      offset += readLength;
    }
    return ret;
  }
}
