package edu.syr.pcpratts.rootbeer.runtime;

public class GpuCard {
	private int cardID;
	private String name;
	private int computeCapabilityVersionA;
	private int computeCapabilityVersionB;
	private int totalMemory; // MBytes
	private int freeMemory; // MBytes
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

	public GpuCard(int cardID, String name, int computeCapabilityVersionA,
			int computeCapabilityVersionB, int totalMemory, int freeMemory,
			int totalRegisterPerBlock, int wrapSize, int maxMemoryPitch,
			int maxThreadsPerBlock, int totalSharedMemoryPerBlock,
			float clockRate, float memoryClockRate, float totalConstantMemory,
			int integrated, int maxThreadsPerMultiprocessor,
			int numberOfMultiprocessor, int maxDimensionXofBlock,
			int maxDimensionYofBlock, int maxDimensionZofBlock,
			int maxDimensionXofGrid, int maxDimensionYofGrid,
			int maxDimensionZofGrid) {
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

	@Override
	public String toString() {
		return "GpuCard [cardID=" + cardID + ", name=" + name + "]";
	}
	
}
