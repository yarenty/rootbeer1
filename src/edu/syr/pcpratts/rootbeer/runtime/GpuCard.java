package edu.syr.pcpratts.rootbeer.runtime;

public class GpuCard {
	private int cardID;
	private String name;
	private int computeCapabilityVersionA;
	private int computeCapabilityVersionB;
	private int totalMemory; // bytes
	private int freeMemory; // bytes
	private int totalRegisterPerBlock;
	private int wrapSize;
	private int maxMemoryPitch;
	private int maxThreadsPerBlock;
	private int totalSharedMemoryPerBlock; // KBytes
	private float clockRate; // MHz
	private float memoryClockRate; // MHz
	private float totalConstantMemory; // MBytes
	private int integrated;
	private int maxThreadsPerMultiprocessor;
	private int numberOfMultiprocessor;
	private int maxDimensionXofBlock;
	private int maxDimensionYofBlock;
	private int maxDimensionZofBlock;
	private int maxDimensionXofGrid;
	private int maxDimensionYofGrid;
	private int maxDimensionZofGrid;

	private long toSpaceAddr;
	private long gpuToSpaceAddr;
	private long textureAddr;
	private long gpuTextureAddr;
	private long gpuClassAddr;
	private long handlesAddr;
	private long gpuHandlesAddr;
	private long exceptionsHandlesAddr;
	private long gpuExceptionsHandlesAddr;
	private long gcInfoSpace;
	private long gpuHeapEndPtr;
	private long gpuBufferSize;

	private long toSpaceSize;
	private long numBlocks;
	private long reserveMem;

	public GpuCard(int cardID, String name, int computeCapabilityVersionA,
			int computeCapabilityVersionB, int totalMemory, int freeMemory,
			int totalRegisterPerBlock, int wrapSize, int maxMemoryPitch,
			int maxThreadsPerBlock, int totalSharedMemoryPerBlock,
			float clockRate, float memoryClockRate, float totalConstantMemory,
			int integrated, int maxThreadsPerMultiprocessor,
			int numberOfMultiprocessor, int maxDimensionXofBlock,
			int maxDimensionYofBlock, int maxDimensionZofBlock,
			int maxDimensionXofGrid, int maxDimensionYofGrid,
			int maxDimensionZofGrid, long toSpaceSize, long numBlocks,
			long reserveMem) {
		super();
		this.cardID = cardID;
		this.name = name;
		this.computeCapabilityVersionA = computeCapabilityVersionA;
		this.computeCapabilityVersionB = computeCapabilityVersionB;
		this.totalMemory = totalMemory;
		this.freeMemory = freeMemory;
		this.totalRegisterPerBlock = totalRegisterPerBlock;
		this.wrapSize = wrapSize;
		this.maxMemoryPitch = maxMemoryPitch;
		this.maxThreadsPerBlock = maxThreadsPerBlock;
		this.totalSharedMemoryPerBlock = totalSharedMemoryPerBlock;
		this.clockRate = clockRate;
		this.memoryClockRate = memoryClockRate;
		this.totalConstantMemory = totalConstantMemory;
		this.integrated = integrated;
		this.maxThreadsPerMultiprocessor = maxThreadsPerMultiprocessor;
		this.numberOfMultiprocessor = numberOfMultiprocessor;
		this.maxDimensionXofBlock = maxDimensionXofBlock;
		this.maxDimensionYofBlock = maxDimensionYofBlock;
		this.maxDimensionZofBlock = maxDimensionZofBlock;
		this.maxDimensionXofGrid = maxDimensionXofGrid;
		this.maxDimensionYofGrid = maxDimensionYofGrid;
		this.maxDimensionZofGrid = maxDimensionZofGrid;
		this.toSpaceSize = toSpaceSize;
		this.numBlocks = numBlocks;
		this.reserveMem = reserveMem;
	}

	public int getCardID() {
		return cardID;
	}

	public String getName() {
		return name;
	}

	public String getComputeCapabilityVersion() {
		return computeCapabilityVersionA + "." + computeCapabilityVersionB;
	}

	public int getTotalMemory() {
		return totalMemory;
	}

	public int getFreeMemory() {
		return freeMemory;
	}

	public int getTotalRegisterPerBlock() {
		return totalRegisterPerBlock;
	}

	public int getWrapSize() {
		return wrapSize;
	}

	public int getMaxMemoryPitch() {
		return maxMemoryPitch;
	}

	public int getMaxThreadsPerBlock() {
		return maxThreadsPerBlock;
	}

	public int getTotalSharedMemoryPerBlock() {
		return totalSharedMemoryPerBlock;
	}

	public float getClockRate() {
		return clockRate;
	}

	public float getMemoryClockRate() {
		return memoryClockRate;
	}

	public float getTotalConstantMemory() {
		return totalConstantMemory;
	}

	public int getIntegrated() {
		return integrated;
	}

	public int getMaxThreadsPerMultiprocessor() {
		return maxThreadsPerMultiprocessor;
	}

	public int getNumberOfMultiprocessor() {
		return numberOfMultiprocessor;
	}

	public int getMaxDimensionXofBlock() {
		return maxDimensionXofBlock;
	}

	public int getMaxDimensionYofBlock() {
		return maxDimensionYofBlock;
	}

	public int getMaxDimensionZofBlock() {
		return maxDimensionZofBlock;
	}

	public int getMaxDimensionXofGrid() {
		return maxDimensionXofGrid;
	}

	public int getMaxDimensionYofGrid() {
		return maxDimensionYofGrid;
	}

	public int getMaxDimensionZofGrid() {
		return maxDimensionZofGrid;
	}

	public long getToSpaceAddr() {
		return toSpaceAddr;
	}

	public long getGpuToSpaceAddr() {
		return gpuToSpaceAddr;
	}

	public long getTextureAddr() {
		return textureAddr;
	}

	public long getGpuTextureAddr() {
		return gpuTextureAddr;
	}

	public long getGpuClassAddr() {
		return gpuClassAddr;
	}

	public long getHandlesAddr() {
		return handlesAddr;
	}

	public long getGpuHandlesAddr() {
		return gpuHandlesAddr;
	}

	public long getExceptionsHandlesAddr() {
		return exceptionsHandlesAddr;
	}

	public long getGpuExceptionsHandlesAddr() {
		return gpuExceptionsHandlesAddr;
	}

	public long getGcInfoSpace() {
		return gcInfoSpace;
	}

	public long getGpuHeapEndPtr() {
		return gpuHeapEndPtr;
	}

	public long getGpuBufferSize() {
		return gpuBufferSize;
	}

	public long getToSpaceSize() {
		return toSpaceSize;
	}

	public long getNumBlocks() {
		return numBlocks;
	}

	public long getReserveMem() {
		return reserveMem;
	}

	@Override
	public String toString() {
		return "GpuCard [cardID=" + cardID + ", name=" + name + "]";
	}

}
