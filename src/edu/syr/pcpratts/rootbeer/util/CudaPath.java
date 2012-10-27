/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package edu.syr.pcpratts.rootbeer.util;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class CudaPath {

  private List<String> m_windowsSearchPaths;
  private List<String> m_unixSearchPaths;
  
  public CudaPath(){
    m_windowsSearchPaths = new ArrayList<String>();
    m_unixSearchPaths = new ArrayList<String>();
    
    m_windowsSearchPaths.add("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\");
    m_windowsSearchPaths.add("C:\\Program Files (x86)\\NVIDIA GPU Computing Toolkit\\CUDA\\");
    m_unixSearchPaths.add("/usr/local/cuda/bin/");
  }
          
  public String get(){
    if(File.separator.equals("/")){
      return getUnix();
    } else {
      return getWindows();
    }
  }

  private String getUnix() {
    if(System.getenv().containsKey("CUDA_BIN_PATH")){
      return System.getenv("CUDA_BIN_PATH");
    }
    for(String path : m_unixSearchPaths){
      File file = new File(path+"nvcc");
      if(file.exists()){
        return path;
      }
    }
    return "/usr/local/cuda/bin/";
  }

  private String getWindows() {
    if(System.getenv().containsKey("CUDA_BIN_PATH")){
      return findWindowsNvcc(System.getenv("CUDA_BIN_PATH"));
    }
    for(String path : m_windowsSearchPaths){
      String nvcc = findWindowsNvcc(path);
      if(nvcc != null){
        return nvcc;
      }
    }
    throw new RuntimeException("cannot find nvcc.exe. Try setting the CUDA_BIN_PATH to the folder with nvcc.exe");
  }

  private String findWindowsNvcc(String path) {
    File file = new File(path);
    File[] children = file.listFiles();
    for(File child : children){
      if(child.isDirectory()){
        findWindowsNvcc(child.getAbsolutePath());
      } else {
        if(child.getName().equals("nvcc.exe")){
          return child.getAbsolutePath();
        }
      }
    }
    return null;
  }
}
