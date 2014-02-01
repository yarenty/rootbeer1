/* 
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 * 
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl.tweaks;

public class NativeCpuTweaks extends Tweaks {

  @Override
  public String getGlobalAddressSpaceQualifier() {
    return "";
  }

  @Override
  public String getUnixHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/UnixNativeHeader.c";
  }
  
  @Override
  public String getWindowsHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/WindowsNativeHeader.c";
  }
  
  @Override
  public String getBothHeaderPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/BothNativeHeader.c";
  }
  
  @Override
  public String getBarrierPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/BarrierNativeBoth.c";
  }

  @Override
  public String getGarbageCollectorPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/GarbageCollector.c";
  }

  @Override
  public String getUnixKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/UnixNativeKernel.c";
  }

  @Override
  public String getWindowsKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/WindowsNativeKernel.c";
  }

  @Override
  public String getBothKernelPath() {
    return "/edu/syr/pcpratts/rootbeer/generate/opencl/BothNativeKernel.c";
  }

  @Override
  public String getDeviceFunctionQualifier() {
    return "";
  }  
}
