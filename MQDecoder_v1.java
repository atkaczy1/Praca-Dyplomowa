/*
 * CVS identifier:
 *
 * $Id: MQDecoder.java,v 1.32 2001/10/17 16:58:00 grosbois Exp $
 *
 * Class:                   MQDecoder
 *
 * Description:             Class that encodes a number of bits using the
 *                          MQ arithmetic decoder
 *
 *
 *
 * COPYRIGHT:
 * 
 * This software module was originally developed by Raphaël Grosbois and
 * Diego Santa Cruz (Swiss Federal Institute of Technology-EPFL); Joel
 * Askelöf (Ericsson Radio Systems AB); and Bertrand Berthelot, David
 * Bouchard, Félix Henry, Gerard Mozelle and Patrice Onno (Canon Research
 * Centre France S.A) in the course of development of the JPEG2000
 * standard as specified by ISO/IEC 15444 (JPEG 2000 Standard). This
 * software module is an implementation of a part of the JPEG 2000
 * Standard. Swiss Federal Institute of Technology-EPFL, Ericsson Radio
 * Systems AB and Canon Research Centre France S.A (collectively JJ2000
 * Partners) agree not to assert against ISO/IEC and users of the JPEG
 * 2000 Standard (Users) any of their rights under the copyright, not
 * including other intellectual property rights, for this software module
 * with respect to the usage by ISO/IEC and Users of this software module
 * or modifications thereof for use in hardware or software products
 * claiming conformance to the JPEG 2000 Standard. Those intending to use
 * this software module in hardware or software products are advised that
 * their use may infringe existing patents. The original developers of
 * this software module, JJ2000 Partners and ISO/IEC assume no liability
 * for use of this software module or modifications thereof. No license
 * or right to this software module is granted for non JPEG 2000 Standard
 * conforming products. JJ2000 Partners have full right to use this
 * software module for his/her own purpose, assign or donate this
 * software module to any third party and to inhibit third parties from
 * using this software module for non JPEG 2000 Standard conforming
 * products. This copyright notice must be included in all copies or
 * derivative works of this software module.
 * 
 * Copyright (c) 1999/2000 JJ2000 Partners.
 * */
package ucar.jpeg.jj2000.j2k.entropy.decoder;

import ucar.jpeg.jj2000.j2k.entropy.encoder.*;

/**
 * This class implements the MQ arithmetic decoder. It is implemented using
 * the software conventions decoder for better performance (i.e. execution
 * time performance). The initial states for each context of the MQ-coder are
 * specified in the constructor.
 * */

// A trick to test for increased speed: merge the Qe and mPS into 1 thing by
// using the sign bit of Qe to signal mPS (positive-or-0 is 0, negative is 1),
// and doubling the Qe, nMPS and nLPS tables. This gets rid of the swicthLM
// table since it can be integrated as special cases in the doubled nMPS and
// nLPS tables. See the JPEG book, chapter 13. The decoded decision can be
// calculated as (q>>>31).

public class MQDecoder {

	/** Length of the range in which current state of the decoder must stay */
    public static final int STATE_RANGE = 1024;
    
    /** The data structures containing the probabilities for the LPS */
    final static
        int qe[]={0x5601, 0x3401, 0x1801, 0x0ac1, 0x0521, 0x0221, 0x5601,
                  0x5401, 0x4801, 0x3801, 0x3001, 0x2401, 0x1c01, 0x1601, 
                  0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401, 0x3001,
                  0x2801, 0x2401, 0x2201, 0x1c01, 0x1801, 0x1601, 0x1401,
                  0x1201, 0x1101, 0x0ac1, 0x09c1, 0x08a1, 0x0521, 0x0441,
                  0x02a1, 0x0221, 0x0141, 0x0111, 0x0085, 0x0049, 0x0025,
                  0x0015, 0x0009, 0x0005, 0x0001, 0x5601 };

    /** The indexes of the next MPS */
    final static
        int nMPS[]={ 1 , 2, 3, 4, 5,38, 7, 8, 9,10,11,12,13,29,15,16,17,
                     18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,
                     35,36,37,38,39,40,41,42,43,44,45,45,46 };

    /** The indexes of the next LPS */
    final static
        int nLPS[]={ 1 , 6, 9,12,29,33, 6,14,14,14,17,18,20,21,14,14,15,
                     16,17,18,19,19,20,21,22,23,24,25,26,27,28,29,30,31,
                     32,33,34,35,36,37,38,39,40,41,42,43,46 };

    /** Whether LPS and MPS should be switched */
    final static
        int switchLM[]={ 1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };

    /** Lookup table for ANS */
    final static int[][][] decoderLookupTable = getDecoderLookupTable();
    
    /** The ByteInputBuffer used to read the compressed bit stream. */
    ByteInputBuffer in;

    /** The current state of the decoder */
    int state;
    
    /** Buffer to hold bits that are about to be decoded */
    public int bitBuffer;
    
    /** Number of bits that are left in bitBuffer */
    public int nrOfBits;
    
    /** The current most probable signal for each context */
    int[] mPS;

    /** The current index of each context */
    int[] I;

    /** 0xFF byte detected */
    boolean ffByteDetected;
    
    /**
     * Instantiates a new MQ-decoder, with the specified number of contexts
     * and initial states. The compressed bytestream is read from the
     * 'iStream' object.
     *
     * @param iStream the stream that contains the coded bits 
     *
     * @param nrOfContexts The number of contexts used
     *
     * @param initStates The initial state for each context. A reference is
     * kept to this array to reinitialize the contexts whenever 'reset()' or
     * 'resetCtxts()' is called.
     * */
    public MQDecoder(ByteInputBuffer iStream,int nrOfContexts,
                     int initStates[]){ 
        in = iStream;
        
        // Default initialization of the statistics bins is MPS=0 and
        // I=0
        I=new int[nrOfContexts];
        mPS=new int[nrOfContexts];
        // Save the initial states

        // Set the contexts
        resetContexts();
        
        // Initialize
        init();

    }

    /**
     * Decodes 'n' symbols from the bit stream using the same context
     * 'ctxt'. If possible the MQ-coder speedup mode will be used to speed up
     * decoding. The speedup mode is used if Q (the LPS probability for 'ctxt'
     * is low enough) and the A and C registers permit decoding several MPS
     * symbols without renormalization.
     *
     * <P>Speedup mode should be used when decoding long runs of MPS with high
     * probability with the same context.
     *
     * <P>This methiod will return the decoded symbols differently if speedup 
     * mode was used or not. If true is returned, then speedup mode was used
     * and the 'n' decoded symbols are all the same and it is returned ain
     * bits[0] only. If false is returned then speedup mode was not used, the
     * decoded symbols are probably not all the same and they are returned in
     * bits[0], bits[1], ... bits[n-1].
     *
     * @param bits The array where to put the decoded symbols. Must be of
     * length 'n' or more.
     *
     * @param ctxt The context to use in decoding the symbols.
     *
     * @param n The number of symbols to decode.
     *
     * @return True if speedup mode was used, false if not. If speedup mode
     * was used then all the decoded symbols are the same and its value is
     * returned in 'bits[0]' only (not in bits[1], bits[2], etc.).
     * */
    public final boolean fastDecodeSymbols(int[] bits, int ctxt, int n) {
    	
    	for (int j = 0; j < n; j++)
        {
        	// Decode symbol
    		int decode[] = decoderLookupTable[I[ctxt]][state - STATE_RANGE];
        	int tempState = decode[1];
        	int shift = decode[2];
        	if (shift > 0)
        		tempState = (tempState << shift) | getBits(shift);
        	int symbol = decode[0] == 1 ? mPS[ctxt] : (1 - mPS[ctxt]);
        	state = tempState;
        	
            bits[j] = symbol;
        }
    	
        return false;
    }
    
    /**
     * This function performs the arithmetic decoding. The function receives 
     * an array in which to put the decoded symbols and an array of contexts 
     * with which to decode them. 
     * 
     * <P>Each context has a current MPS and an index describing what the
     * current probability is for the LPS. Each bit is decoded and if the
     * probability of the LPS exceeds .5, the MPS and LPS are switched.
     *
     * @param bits The array where to place the decoded symbols. It should be
     * long enough to contain 'n' elements.
     *
     * @param cX The context to use in decoding each symbol.
     *
     * @param n The number of symbols to decode
     * */
    public final void decodeSymbols(int[] bits, int[] cX, int n){
        for (int j = 0; j < n; j++)
        {
        	// Decode symbol
        	int[] decode = decoderLookupTable[I[cX[j]]][state - STATE_RANGE];
        	int tempState = decode[1];
        	int shift = decode[2];
        	if (shift > 0)
        		tempState = (tempState << shift) | getBits(shift);
        	int symbol = decode[0] == 1 ? mPS[cX[j]] : (1 - mPS[cX[j]]);
        	state = tempState;
        	
            bits[j] = symbol;
        }
    }


    /**
     * Arithmetically decodes one symbol from the bit stream with the given
     * context and returns its decoded value.
     *
     * <P>Each context has a current MPS and an index describing what the
     * current probability is for the LPS. Each bit is encoded and if the
     * probability of the LPS exceeds .5, the MPS and LPS are switched.
     *
     * @param context The context to use in decoding the symbol
     *
     * @return The decoded symbol, 0 or 1.
     * */
    public final int decodeSymbol(int context){
    	
    	// Decode symbol
    	int[] decode = decoderLookupTable[I[context]][state - STATE_RANGE];
    	int tempState = decode[1];
    	int shift = decode[2];
    	if (shift > 0)
    		tempState = (tempState << shift) | getBits(shift);
    	int symbol = decode[0] == 1 ? mPS[context] : (1 - mPS[context]);
    	state = tempState;
    	return symbol;
    }

    /**
     * Checks for past errors in the decoding process using the predictable
     * error resilient termination. This works only if the encoder used the
     * predictable error resilient MQ termination, otherwise it reports wrong
     * results. If an error is detected it means that the MQ bit stream has
     * been wrongly decoded or that the MQ terminated segment length is too
     * long. If no errors are detected it does not necessarily mean that the
     * MQ bit stream has been correctly decoded.
     *
     * @return True if errors are found, false otherwise.
     * */
    public boolean checkPredTerm() {
        return false;
    }

    /**
     * Returns the number of contexts in the arithmetic coder.
     *
     * @return The number of contexts
     **/
    public final int getNumCtxts(){
        return I.length;
    }

    /**
     * Resets a context to the original probability distribution.
     *
     * @param c The number of the context (it starts at 0).
     * */
    public final void resetCtxt(int c){
    	int read = in.read();
        I[c] = read & 0x7F;
        mPS[c] = (read >> 7) & 1;
    }

    /**
     * Resets a context to the original probability distribution. The original
     * probability distribution depends on the actual implementation of the
     * arithmetic coder or decoder.
     *
     * @param c The index of the context (it starts at 0).
     * */
    private final void resetContexts(){
        for (int c = 0; c < I.length; c++)
        {
        	int read = in.read();
            I[c] = read & 0x7F;
            mPS[c] = (read >> 7) & 1;
        }
    }
    
    /**
     * Resets a context to the original probability distribution. The original
     * probability distribution depends on the actual implementation of the
     * arithmetic coder or decoder.
     *
     * @param c The index of the context (it starts at 0).
     * */
    public final void resetCtxts(){
    	// EMPTY
    }

    /**
     * Resets the MQ decoder to start a new segment. This is like recreating a
     * new MQDecoder object with new input data.
     *
     * @param buf The byte array containing the MQ encoded data. If null the
     * current byte array is assumed.
     *
     * @param off The index of the first element in 'buf' to be decoded. If
     * negative the byte just after the previous segment is assumed, only
     * valid if 'buf' is null.
     *
     * @param len The number of bytes in 'buf' to be decoded. Any subsequent
     * bytes are taken to be 0xFF.
     * */
    public final void nextSegment(byte buf[], int off, int len) {
        // Set the new input
        in.setByteArray(buf,off,len);
        // Reinitialize MQ
        resetContexts();
        init();
    }

    /**
     * Returns the underlying 'ByteInputBuffer' from where the MQ coded input
     * bytes are read.
     *
     * @return The underlying ByteInputBuffer.
     * */
    public ByteInputBuffer getByteInputBuffer() {
        return in;
    }

    /**
     * Initializes the state of the MQ coder, without modifying the current
     * context states. It sets the registers (A,C,B) and the "marker found"
     * state to the initial state, to start the decoding of a new segment.
     *
     * <P>To have a complete reset of the MQ (as if a new MQDecoder object was
     * created) 'resetCtxts()' should be called after this method.
     * */
    private void init() {
        // --- INITDEC

    	ffByteDetected = false;
    	
        // Read data to bitBuffer
        bitBuffer = 0;
        nrOfBits = 0;
        for (int i = 0; i < 4; i++)
        {
        	int readByte = in.read();
        	if (readByte == -1)
        		break;
        	nrOfBits += 8;
        	bitBuffer |= readByte << ((3 - i) * 8);
        }
        dropExtraBits();
        
        int rangeBitCount = 0;
        int rangeLowerBound = STATE_RANGE;
        while(rangeLowerBound > 0)
        {
        	rangeBitCount++;
        	rangeLowerBound >>>= 1;
        }
        
        // Read the initial state of the decoder
        state = (bitBuffer >> (32 - rangeBitCount)) & ((1 << rangeBitCount) - 1);
        bitBuffer <<= rangeBitCount;
        nrOfBits -= rangeBitCount;

        // End of INITDEC ---
    }
    
    /**
     * Returns decoder lookup tables created based on coding table
     * @return
     */
    public static int[][][] getDecoderLookupTable()
    {
    	int[][][] decoderLookupTable = new int[47][STATE_RANGE][3];
    	int[][][][] coderLookupTable = MQCoder.coderLookupTable;
    	for (int i = 0; i < 47; i++)
    	{
    		for (int j = 0; j < STATE_RANGE; j++)
    		{
    			for (int k = 0; k < 2; k++)
    			{
    				int nextState = coderLookupTable[i][j][k][2] - STATE_RANGE;
    				int nrOfShifted = coderLookupTable[i][j][k][1];
    				decoderLookupTable[i][nextState][0] = k;
    				decoderLookupTable[i][nextState][1] = (j + STATE_RANGE) >> nrOfShifted;
    				decoderLookupTable[i][nextState][2] = nrOfShifted;
    			}
    		}
    	}
    	
    	
    	return decoderLookupTable;
    }
    
    /**
     * Deletes extra 0 bits after 0xFF bytes
     */
    private void dropExtraBits()
    {
    	int tempBitBuffer = 0; // bit buffer after deleting extra bits
    	int tempBitBufferCount = 0; // Number of bits written to tempBitBuffer
    	
    	// Iterates over bytes in the bit buffer
    	for (int i = 3; i >= 0; i--)
    	{
    		int offset = i * 8; // Offset of the byte in the bit buffer
    		
    		// Check if previous byte was 0xFF
    		if (ffByteDetected == true)
        	{
    			// Add all the bits except extra one to the bit buffer
        		int tempA = (bitBuffer << 1) & (0xFE << offset);
        		tempBitBuffer |= tempA << ((24 - offset) - tempBitBufferCount);
        		tempBitBufferCount += 7;
        	}
        	else
        	{
        		// Add all the bits to the bit buffer
        		int tempA = bitBuffer & (0xFF << offset);
        		tempBitBuffer |= tempA << ((24 - offset) - tempBitBufferCount);
        		tempBitBufferCount += 8;
        	}
    		
    		// Check if the current byte == 0xFF
    		if (((bitBuffer >>> offset) & 0xFF) == 0xFF)
    		{
    			ffByteDetected = true;
    		}
    		else
    		{
    			ffByteDetected = false;
    		}
    	}
    	
    	bitBuffer = tempBitBuffer;
    	nrOfBits = tempBitBufferCount;
    }
    
    /**
     * Returns bits from the input
     * @param count - number of bits to return
     * @return
     */
    private int getBits(int count)
    {
    	int countToRead = count;
    	int bits = 0;
    	
    	// Go in a loop until all wanted bits has been read
    	while(countToRead > 0)
    	{
    		// Calculate number of bits that can be read in the current iteration
    		int c = countToRead < nrOfBits ? countToRead : nrOfBits;
    		
    		int b = bitBuffer >>> (32 - c);
    		bitBuffer <<= c;
    		nrOfBits -= c;
    		countToRead -= c;
    		bits = (bits << c) | b;
    		
    		// If bit buffer is already empty, fill it
    		if (nrOfBits == 0)
    		{
    			for (int i = 0; i < 4; i++)
    	        {
    	        	int readByte = in.read();
    	        	if (readByte == -1)
    	        		break;
    	        	nrOfBits += 8;
    	        	bitBuffer |= readByte << ((3 - i) << 3);
    	        }
    			
    			// Delete extra 0 bits after every 0xFF byte
    	        dropExtraBits();
    		}
    	}
    	return bits;
    }
}