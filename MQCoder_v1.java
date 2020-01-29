/*
 * CVS identifier:
 *
 * $Id: MQCoder.java,v 1.36 2002/01/10 10:31:28 grosbois Exp $
 *
 * Class:                   MQCoder
 *
 * Description:             Class that encodes a number of bits using the
 *                          MQ arithmetic coder
 *
 *
 *                          Diego SANTA CRUZ, Jul-26-1999 (improved speed)
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
package ucar.jpeg.jj2000.j2k.entropy.encoder;

import java.util.*;

import ucar.jpeg.jj2000.j2k.util.ArrayUtil;

/**
 * This class implements the MQ arithmetic coder. When initialized a specific
 * state can be specified for each context, which may be adapted to the
 * probability distribution that is expected for that context.
 *
 * <p>The type of length calculation and termination can be chosen at
 * construction time.
 * 
 * ---- Tricks that have been tried to improve speed ----
 *
 * <p>1) Merging Qe and mPS and doubling the lookup tables<br>
 *
 * Merge the mPS into Qe, as the sign bit (if Qe>=0 the sense of MPS is 0, if
 * Qe<0 the sense is 1), and double the lookup tables. The first half of the
 * lookup tables correspond to Qe>=0 (i.e. the sense of MPS is 0) and the
 * second half to Qe<0 (i.e. the sense of MPS is 1). The nLPS lookup table is
 * modified to incorporate the changes in the sense of MPS, by making it jump
 * from the first to the second half and vice-versa, when a change is
 * specified by the swicthLM lookup table. See JPEG book, section 13.2, page
 * 225. <br>
 *
 * There is NO speed improvement in doing this, actually there is a slight
 * decrease, probably due to the fact that often Q has to be negated. Also the
 * fact that a brach of the type "if (bit==mPS[li])" is replaced by two
 * simpler braches of the type "if (bit==0)" and "if (q<0)" may contribute to
 * that.</p>
 *
 * <p>2) Removing cT<br>
 *
 * It is possible to remove the cT counter by setting a flag bit in the high
 * bits of the C register. This bit will be automatically shifted left
 * whenever a renormalization shift occurs, which is equivalent to decreasing
 * cT. When the flag bit reaches the sign bit (leftmost bit), which is
 * equivalenet to cT==0, the byteOut() procedure is called. This test can be
 * done efficiently with "c<0" since C is a signed quantity. Care must be
 * taken in byteOut() to reset the bit in order to not interfere with other
 * bits in the C register. See JPEG book, page 228.<br>
 *
 * There is NO speed improvement in doing this. I don't really know why since
 * the number of operations whenever a renormalization occurs is
 * decreased. Maybe it is due to the number of extra operations in the
 * byteOut(), terminate() and getNumCodedBytes() procedures.</p>
 *
 * <p>3) Change the convention of MPS and LPS.<br>
 *
 * Making the LPS interval be above the MPS interval (MQ coder convention is
 * the opposite) can reduce the number of operations along the MPS path. In
 * order to generate the same bit stream as with the MQ convention the output
 * bytes need to be modified accordingly. The basic rule for this is that C =
 * (C'^0xFF...FF)-A, where C is the codestream for the MQ convention and C' is
 * the codestream generated by this other convention. Note that this affects
 * bit-stuffing as well.<br>
 *
 * This has not been tested yet.<br>
 *
 * <p>4) Removing normalization while loop on MPS path<br>
 *
 * Since in the MPS path Q is guaranteed to be always greater than 0x4000
 * (decimal 0.375) it is never necessary to do more than 1 renormalization
 * shift. Therefore the test of the while loop, and the loop itself, can be
 * removed.</p>
 *
 * <p>5) Simplifying test on A register<br>
 *
 * Since A is always less than or equal to 0xFFFF, the test "(a & 0x8000)==0"
 * can be replaced by the simplete test "a < 0x8000". This test is simpler in
 * Java since it involves only 1 operation (although the original test can be
 * converted to only one operation by  smart Just-In-Time compilers)<br>
 *
 * This change has been integrated in the decoding procedures.</p>
 *
 * <p>6) Speedup mode<br>
 *
 * Implemented a method that uses the speedup mode of the MQ-coder if
 * possible. This should greately improve performance when coding long runs of 
 * MPS symbols that have high probability. However, to take advantage of this, 
 * the entropy coder implementation has to explicetely use it. The generated
 * bit stream is the same as if no speedup mode would have been used.<br>
 *
 * Implemented but performance not tested yet.</p>
 *
 * <p>7) Multiple-symbol coding<br>
 *
 * Since the time spent in a method call is non-negligable, coding several
 * symbols with one method call reduces the overhead per coded symbol. The
 * decodeSymbols() method implements this. However, to take advantage of it,
 * the implementation of the entropy coder has to explicitely use it.<br>
 *
 * Implemented but performance not tested yet.</p>
 *  */
public class MQCoder {

    /** Identifier for the lazy length calculation. The lazy length
     * calculation is not optimal but is extremely simple. */
    public static final int LENGTH_LAZY = 0;

    /** Identifier for a very simple length calculation. This provides better
     * results than the 'LENGTH_LAZY' computation. This is the old length
     * calculation that was implemented in this class. */
    public static final int LENGTH_LAZY_GOOD = 1;

    /** Identifier for the near optimal length calculation. This calculation
     * is more complex than the lazy one but provides an almost optimal length 
     * calculation. */
    public static final int LENGTH_NEAR_OPT = 2;

    /** The identifier fort the termination that uses a full flush. This is
     * the less efficient termination. */
    public static final int TERM_FULL = 0;

    /** The identifier for the termination that uses the near optimal length
     * calculation to terminate the arithmetic codeword */
    public static final int TERM_NEAR_OPT = 1;

    /** The identifier for the easy termination that is simpler than the
     * 'TERM_NEAR_OPT' one but slightly less efficient. */
    public static final int TERM_EASY = 2;

    /** The identifier for the predictable termination policy for error
     * resilience. This is the same as the 'TERM_EASY' one but an special
     * sequence of bits is embodied in the spare bits for error resilience
     * purposes. */
    public static final int TERM_PRED_ER = 3;

    /** Length of the range in which current state of the coder must stay */
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
    final static        // at indices 0, 6, and 14 we switch
        int switchLM[]={ 1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,
                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    // Having ints proved to be more efficient than booleans
    
    /** Array of probabilities value for all 47 probability models */
    static double[] probabilities;
    
    /** Array of limits for ANS coder. While encoding a symbol, coder shifts bits
     * out of the state until the state is low enough that after next encoding
     * state will stay between STATE_RANGE and 2 * STATE_RANGE - 1. 
     * First array dimension - probability model
     * Second array dimension - symbol that is about to be encoded */
    static int[][] limits;
    
    /** Init probabilities and limits */
    static
    {
    	probabilities = new double[47];
    	limits = new int[47][2];
    	getProbabilitiesAndLimits(probabilities, limits);
    }
    
    /** Lookup table for ANS */
    public final static int[][][][] coderLookupTable = getCoderLookupTable();
    
    /** The ByteOutputBuffer used to write the compressed bit stream. */
    ByteOutputBuffer out;
    
    /** The ByteOutputBuffer used to buffer compressed bit stream before reversing it. */
    ByteOutputBuffer outBuffer;

    /** Buffer for input symbols */
    List<Integer> symbolBuffer;
    
    /** Buffer for input contexts */
    List<Integer> contextBuffer;
    
    /** Current state of the ANS coder */
    int state;
    
    /** Buffer for bits that are going to be transfered */
    int bitsBuffer;
    
    /** The number of bits stored in bitsBuffer */
    int nrOfBits;
    
    /** The current most probable signal for each context */
    int[] mPS;

    /** The current index of each context */
    int[] I;

    /** The initial state of each context */
    int initStates[];
    
    /** Array of number of bits coded in every context */
    int[] totalCount;
    
    /** Array of number of 1 bits coded in every context */
    int[] oneCount;
    
    /** 
     * Set the length calculation type to the specified type.
     *
     * @param ltype The type of length calculation to use. One of
     * 'LENGTH_LAZY', 'LENGTH_LAZY_GOOD' or 'LENGTH_NEAR_OPT'.
     * */
    public void setLenCalcType(int ltype) {
        // EMPTY
    }

    /** 
     * Set termination type to the specified type.
     *
     * @param ttype The type of termination to use. One of 'TERM_FULL',
     * 'TERM_NEAR_OPT', 'TERM_EASY' or 'TERM_PRED_ER'.
     * */
    public void setTermType(int ttype) {
        // EMPTY
    }

    /**
     * Instantiates a new MQ-coder, with the specified number of contexts and
     * initial states. The compressed bytestream is written to the 'oStream'
     * object.
     *
     * @param oStream where to output the compressed data.
     *
     * @param nrOfContexts The number of contexts used by the MQ coder.
     *
     * @param init The initial state for each context. A reference is kept to
     * this array to reinitialize the contexts whenever 'reset()' or
     * 'resetCtxts()' is called.
     * */
    public MQCoder(ByteOutputBuffer oStream, int nrOfContexts, int init[]) {
        out = oStream;

        // --- INITENC

        // Default initialization of the statistics bins is MPS=0 and
        // I=0
        I=new int[nrOfContexts];
        mPS=new int[nrOfContexts];
        totalCount = new int[nrOfContexts];
        oneCount = new int[nrOfContexts];
        initStates = init;

        state = STATE_RANGE;
        
        outBuffer = new ByteOutputBuffer();
        symbolBuffer = new ArrayList<>();
        contextBuffer = new ArrayList<>();
        
        bitsBuffer = 0;
        nrOfBits = 0;
        
        resetCtxts();        

        
    }

    /**
     * This method performs the coding of the symbol 'bit', using context
     * 'ctxt', 'n' times, using the MQ-coder speedup mode if possible.
     *
     * <p>If the symbol 'bit' is the current more probable symbol (MPS) and
     * qe[ctxt]<=0x4000, and (A-0x8000)>=qe[ctxt], speedup mode will be
     * used. Otherwise the normal mode will be used. The speedup mode can
     * significantly improve the speed of arithmetic coding when several MPS
     * symbols, with a high probability distribution, must be coded with the
     * same context. The generated bit stream is the same as if the normal mode
     * was used.</p>
     *
     * <p>This method is also faster than the 'codeSymbols()' and
     * 'codeSymbol()' ones, for coding the same symbols with the same context
     * several times, when speedup mode can not be used, although not
     * significantly.</p>
     *
     * @param bit The symbol do code, 0 or 1.
     *
     * @param ctxt The context to us in coding the symbol.
     *
     * @param n The number of times that the symbol must be coded.
     * */
    public final void fastCodeSymbols(int bit, int ctxt, int n) {
        for (int i = 0; i < n; i++)
        {
        	// Buffer symbol and context
        	symbolBuffer.add(bit);
        	contextBuffer.add(ctxt);
        	
        	// Update statistics
        	totalCount[ctxt]++;
        	if (bit == 1)
        		oneCount[ctxt]++;
        	
        }
    }
    
    /**
     * This function performs the arithmetic encoding of several symbols
     * together. The function receives an array of symbols that are to be
     * encoded and an array containing the contexts with which to encode them.
     *
     * <p>The advantage of using this function is that the cost of the method
     * call is amortized by the number of coded symbols per method call.</p>
     * 
     * <p>Each context has a current MPS and an index describing what the 
     * current probability is for the LPS. Each bit is encoded and if the
     * probability of the LPS exceeds .5, the MPS and LPS are switched.</p>
     *
     * @param bits An array containing the symbols to be encoded. Valid
     * symbols are 0 and 1.
     *
     * @param cX The context for each of the symbols to be encoded.
     *
     * @param n The number of symbols to encode.
     * */
    public final void codeSymbols(int[] bits, int[] cX, int n) {
    	
    	for (int i = 0; i < n; i++)
        {
    		// Buffer symbol and context
        	symbolBuffer.add(bits[i]);
        	contextBuffer.add(cX[i]);
        	
        	// Update statistics
        	totalCount[cX[i]]++;
        	if (bits[i] == 1)
        		oneCount[cX[i]]++;
        	
        }
    }


    /**
     * This function performs the arithmetic encoding of one symbol. The 
     * function receives a bit that is to be encoded and a context with which
     * to encode it.
     *
     * <p>Each context has a current MPS and an index describing what the 
     * current probability is for the LPS. Each bit is encoded and if the
     * probability of the LPS exceeds .5, the MPS and LPS are switched.</p>
     *
     * @param bit The symbol to be encoded, must be 0 or 1.
     *
     * @param context the context with which to encode the symbol.
     * */
    public final void codeSymbol(int bit, int context) {
    	
    	// Buffer symbol and context
        symbolBuffer.add(bit);
        contextBuffer.add(context);
        
        // Update statistics
        totalCount[context]++;
    	if (bit == 1)
    		oneCount[context]++;
    	
    }
    
    /**
     * This function flushes the remaining encoded bits and makes sure that
     * enough information is written to the bit stream to be able to finish
     * decoding, and then it reinitializes the internal state of the MQ coder
     * but without modifying the context states.
     *
     * <p>After calling this method the 'finishLengthCalculation()' method
     * should be called, after compensating the returned length for the length
     * of previous coded segments, so that the length calculation is
     * finalized.</p>
     *
     * <p>The type of termination used depends on the one specified at the
     * constructor.</p>
     *
     * @return The length of the arithmetic codeword after termination, in
     * bytes.
     * */
    public int terminate() {
    	
    	// Set probability models for every context according to collected statistics */
    	for (int ctxt = 0; ctxt < I.length; ctxt++)
    	{
    		// It doesn't update state of the uniform context
    		if (I[ctxt] != 46)
        	{
    			// Calculate LPS probabilities and set MPS for a context
        		double p;
        		if (totalCount[ctxt] == 0)
        		{
        			mPS[ctxt] = 0;
        			p = 0;
        		}
        		else
        		{
	        		if ((oneCount[ctxt] << 1) > totalCount[ctxt])
	        		{
	        			mPS[ctxt] = 1;
	        			p = ((double) (totalCount[ctxt] - oneCount[ctxt])) / ((double) totalCount[ctxt]);
	        		}
	        		else
	        		{
	        			mPS[ctxt] = 0;
	        			p = ((double) oneCount[ctxt]) / ((double) totalCount[ctxt]);
	        		}
        		}
        		
        		// Find the probability model that is the closest to counted probability
        		int minState = -1;
        		double minDifference = 1.1;
        		for (int j = 0; j < 46; j++)
        		{
        			double diff = probabilities[j] - p;
        			diff = diff > 0 ? diff : -diff;
        			if (diff < minDifference)
        			{
        				minDifference = diff;
        				minState = j;
        			}
        		}
        		
        		I[ctxt] = minState;
        	}
    	}
    	
    	// Encoding of the all buffered symbols. Encoding is performed in reverse order
        for (int i = symbolBuffer.size() - 1; i >= 0; i--)
        {
        	int s = symbolBuffer.get(i);
        	int c = contextBuffer.get(i);
        	
        	int sym = (s == mPS[c] ? 1 : 0);
        	
        	int[] code = coderLookupTable[I[c]][state - STATE_RANGE][sym];
        	
        	int nos = code[1];
        	int sh = code[0];
        	state = code[2];
        	
        	// Output bits if needed
        	if (nos > 0)
        	{
        		bitsOut(sh, nos);
        	}
        	
        }
        
        // Flushes state and bit buffer to output stream
        flushToOutput();
        
        // Add states to code stream
        for (int i = 0; i < I.length; i++)
        {
        	out.write(I[i] | (mPS[i] << 7));
        }
        
        // Reverse bit stream and shift it to the beginning of the first byte
        byte[] byteBuffer = getOutputByteArray();
        
        // Add additional 0 bit after every 0xFF byte
        int nrOfWrittenBytes = addAdditionalBits(byteBuffer) + I.length;
    	
        return nrOfWrittenBytes;
    }
    
    /**
     * Writes bits to the output
     * @param bits - bits that are about to be placed in the output
     * @param count - number of bits placed in the output
     */
    private void bitsOut(int bits, int count)
    {
    	int nrOfBitsToWrite = count;
    	int bitsToWrite = bits;
    	
    	// Go in the loop until all bits has been written
    	while(nrOfBitsToWrite > 0)
    	{
    		// Calculate number of bits that can be written in this iteration
    		int writtenInCurrent = 32 - nrOfBits < nrOfBitsToWrite ? 32 - nrOfBits : nrOfBitsToWrite;
    		
    		// Write bits to bit buffer
    		bitsBuffer >>>= writtenInCurrent;
    		bitsBuffer |= bitsToWrite << (32 - writtenInCurrent);
    		
    		bitsToWrite >>>= writtenInCurrent;
    		nrOfBits += writtenInCurrent;
    		nrOfBitsToWrite -= writtenInCurrent;
    		
    		// If the bit buffer is full flush it to output
    		if (nrOfBits == 32)
    		{
    			for (int i = 0; i < 4; i++)
    			{
    				outBuffer.write((bitsBuffer >>> (i * 8)) & 0xFF);
    			}
    			nrOfBits = 0;
    			bitsBuffer = 0;
    		}
    	}
    }

    /**
     * Returns the number of contexts in the arithmetic coder.
     *
     * @return The number of contexts
     * */
    public final int getNumCtxts() {
        return I.length;
    }

    /**
     * Resets a context to the original probability distribution, and sets its
     * more probable symbol to 0.
     *
     * @param c The number of the context (it starts at 0).
     * */
    public final void resetCtxt(int c) {
        I[c]=initStates[c];
        mPS[c] = 0;
        totalCount[c] = 0;
        oneCount[c] = 0;
    }

    /**
     * Resets all contexts to their original probability distribution and sets
     * all more probable symbols to 0.
     * */
    public final void resetCtxts() {
        System.arraycopy(initStates,0,I,0,I.length);
        ArrayUtil.intArraySet(mPS,0);
        ArrayUtil.intArraySet(totalCount,0);
        ArrayUtil.intArraySet(oneCount,0);
    }

    /**
     * Returns the number of bytes that are necessary from the compressed
     * output stream to decode all the symbols that have been coded this
     * far. The number of returned bytes does not include anything coded
     * previous to the last time the 'terminate()' or 'reset()' methods where
     * called.
     *
     * <p>The values returned by this method are then to be used in finishing
     * the length calculation with the 'finishLengthCalculation()' method,
     * after compensation of the offset in the number of bytes due to previous
     * terminated segments.</p>
     *
     * <p>This method should not be called if the current coding pass is to be
     * terminated. The 'terminate()' method should be called instead.</p>
     *
     * <p>The calculation is done based on the type of length calculation
     * specified at the constructor.</p>
     *
     * @return The number of bytes in the compressed output stream necessary
     * to decode all the information coded this far.
     * */
    public final int getNumCodedBytes() {
        return 0;
    }

    /**
     * Reinitializes the MQ coder and the underlying 'ByteOutputBuffer' buffer
     * as if a new object was instantaited. All the data in the
     * 'ByteOutputBuffer' buffer is erased and the state and contexts of the
     * MQ coder are reinitialized). Additionally any saved MQ states are
     * discarded.
     * */
    public final void reset() {

        // Reset the output buffer
        out.reset();
        outBuffer.reset();
        symbolBuffer.clear();
        contextBuffer.clear();
        
        state = STATE_RANGE;
        bitsBuffer = 0;
        nrOfBits = 0;
        
        resetCtxts();
    }
    
    /**
     * Terminates the calculation of the required length for each coding
     * pass. This method must be called just after the 'terminate()' one has
     * been called for each terminated MQ segment.
     *
     * <p>The values in 'rates' must have been compensated for any offset due
     * to previous terminated segments, so that the correct index to the
     * stored coded data is used.</p>
     *
     * @param rates The array containing the values returned by
     * 'getNumCodedBytes()' for each coding pass.
     *
     * @param n The index in the 'rates' array of the last terminated length.
     * */
    public void finishLengthCalculation(int rates[],int n) {
        for (int i = n - 1; i >= 0; i--)
        {
        	rates[i] = rates[n];
        }
    }
    
    /** Get whole encoding lookup table
     * 
     * @return
     */
    public static int[][][][] getCoderLookupTable()
    {
    	int[][][][] lookupTable =  new int[47][STATE_RANGE][2][3];
    	
    	for (int i = 0; i < 46; i++)
    	{
    		lookupTable[i] = getCoderLookupTableForProbability(probabilities[i], limits[i]);
    	}
    	
    	// Last lookup table has the same probabilities as the first one
    	lookupTable[46] = lookupTable[0];
    	
    	return lookupTable;
    }
    
    /**
     * Fill the arrays containing probabilities and limits used for all probabilities
     * models. For very low probabilities there are less than 2 states assigned to
     * the LPS therefore no limit value can be found. To solve this problem low
     * probabilities are risen to the last correct probability
     * @param probabilities
     * @param limits
     */
    public static void getProbabilitiesAndLimits(double[] probabilities, int[][] limits)
    {
    	double wholeRange = qe[0] * 2;
    	double last_correct_probability = 0.5;
    	int[] last_correct_limits = null;
    	boolean values_incorrect = false;
    	for (int i = 0; i < 47; i++)
    	{
    		if (!values_incorrect)
    		{
    			double temp_prob = qe[i] / wholeRange;
    			int[] temp_limits = getLimits(temp_prob);
    			if (temp_limits[0] < 1)
    			{
    				probabilities[i] = last_correct_probability;
    				limits[i] = last_correct_limits;
    				values_incorrect = true;
    			}
    			else
    			{
    				probabilities[i] = last_correct_probability = temp_prob;
    				limits[i] = last_correct_limits = temp_limits;
    			}
    		}
    		else
    		{
    			probabilities[i] = last_correct_probability;
				limits[i] = last_correct_limits;
    		}
    	}
    }
    
    /** Get lookup table for the specified probability
     * 
     * @param p
     * @return
     */
    private static int[][][] getCoderLookupTableForProbability(double p, int[] limits)
    {
    	int[][][] lookupTable = new int[STATE_RANGE][2][3];
    	int state;
    	int shiftedBits;
		int nrOfShifted;
    	
    	for (int i = 0; i < STATE_RANGE; i++)
    	{
    		// LPS
    		state = i + STATE_RANGE;
    		shiftedBits = 0;
    		nrOfShifted = 0;
    		while(state > limits[0])
    		{
    			shiftedBits += (state & 1) << nrOfShifted;
    			nrOfShifted++;
    			state >>= 1;
    		}
    		state = coderZeroEquation(state, p);
    		lookupTable[i][0][0] = shiftedBits;
    		lookupTable[i][0][1] = nrOfShifted;
    		lookupTable[i][0][2] = state;
    		
    		// MPS
    		state = i + STATE_RANGE;
    		shiftedBits = 0;
    		nrOfShifted = 0;
    		while(state > limits[1])
    		{
    			shiftedBits += (state & 1) << nrOfShifted;
    			nrOfShifted++;
    			state >>= 1;
    		}
    		state = coderOneEquation(state, p);
    		lookupTable[i][1][0] = shiftedBits;
    		lookupTable[i][1][1] = nrOfShifted;
    		lookupTable[i][1][2] = state;
    	}
    	
    	lookupTable = lookupTableCorrection(lookupTable);
    	return lookupTable;
    }
    
    /** Get limiting states for both LPS and MPS, so that next state will be in range
     * <STATE_RANGE, 2 * STATE_RANGE - 1>
     * @param p - probability for which calculate limits
     * @return
     */
    private static int[] getLimits(double p)
    {
    	int[] limits = new int[2];
    	limits[0] = (int) ((2 * STATE_RANGE) * p);
    	limits[1] = (int) ((2 * STATE_RANGE) * (1.0 - p));
    	
    	if (coderZeroEquation(limits[0], p) >= 2 * STATE_RANGE)
    		limits[0]--;
    	if (coderOneEquation(limits[1], p) >= 2 * STATE_RANGE)
    		limits[1]--;
    	
    	return limits;
    }
    
    /**
     * Encoding equation for symbol 0
     * @param state
     * @param probability
     * @return
     */
    private static int coderZeroEquation(int state, double probability)
    {
    	int result = (int)(Math.ceil(((double) (state + 1)) / probability)) - 1;
    	return result;
    }
    
    /**
     * Encoding equation for symbol 1
     * @param state
     * @param probability
     * @return
     */
    private static int coderOneEquation(int state, double probability)
    {
    	int result = (int) Math.floor(((double) state) / (1.0 - probability));
    	return result;
    }
    
    /**
     * Correction to lookup tables generated through equations. Some of the state
     * transitions point to the state STATE_RANGE - 1
     * @param lookupTable
     * @return
     */
    private static int[][][] lookupTableCorrection(int[][][] lookupTable)
    {
    	for (int i = 0; i < STATE_RANGE; i++)
    	{
    		// Check if the table needs correction
    		if (lookupTable[i][1][2] < STATE_RANGE)
    		{
    			// Change values in incorrect transition
    			lookupTable[i][1][0] = 0;
    			lookupTable[i][1][1] = 0;
    			lookupTable[i][1][2] = 2 * STATE_RANGE - 1;
    			
    			// Change values in previous transitions
    			int expectedState = 2 * STATE_RANGE - 1;
    			int state = i - 1;
    			while(state >= 0)
    			{
    				if (lookupTable[state][1][2] == expectedState)
    				{
    					expectedState--;
    					lookupTable[state][1][2] = expectedState;
    					state--;
    				}
    				else
    				{
    					break;
    				}
    			}
    			
    			// Change values in proper transitions in 0 coding table
    			for (int j = 0; j < STATE_RANGE; j++)
    			{
    				if (lookupTable[j][0][2] == expectedState)
    				{
    					int k;
    					for (k = j + 1; lookupTable[k][0][2] == expectedState; k++)
    					{
    						// Empty iteration
    					}
    					for (int replaceIter = j; replaceIter < k; replaceIter++)
    					{
    						lookupTable[replaceIter][0][2] = lookupTable[k][0][2];
    						lookupTable[replaceIter][0][1] = lookupTable[k][0][1];
    					}
    					break;
    				}
    			}
    			break;
    		}
    	}
    	return lookupTable;
    }
    
    /**
     * Flushes state and bit buffer to output
     */
    private void flushToOutput()
    {
    	int rangeBitCount = 0;
        int rangeLowerBound = STATE_RANGE;
        while(rangeLowerBound > 0)
        {
        	rangeBitCount++;
        	rangeLowerBound >>>= 1;
        }
    	
    	// Flush state to the bitBuffer
        bitsOut(state, rangeBitCount);
        
        if (nrOfBits % 8 != 0)
        {
        	bitsBuffer >>>= (8 - (nrOfBits % 8));
        }
        
        // Flush bitBuffer to outBuffer
        int nrOfFlushed = nrOfBits / 8;
        if (nrOfBits % 8 != 0)
        	nrOfFlushed++;
        
        while(nrOfFlushed > 0)
        {
        	outBuffer.write((bitsBuffer >> ((4 - nrOfFlushed) * 8)) & 0xFF);
        	nrOfFlushed--;
        }
        bitsBuffer = 0;
    }
    
    /**
     * Add extra 0 bits after every 0xFF byte
     * @param output
     * @return
     */
    private int addAdditionalBits(byte[] output)
    {
    	int nrOfWrittenBytes = 0;
    	
    	int nrOfCarried = 0;
        int carry = 0;
        for (int i = 0; i < output.length; i++)
        {
        	int temp = carry | ((output[i] & 0xFF) >>> nrOfCarried);
        	carry = (output[i] << (8 - nrOfCarried)) & 0xFF;
        	if (temp == 0xFF)
        	{
        		nrOfCarried++;
        		carry = (carry >>> 1) & 0x7F;
        	}
        	
        	nrOfWrittenBytes++;
        	out.write(temp);
        	if (nrOfCarried == 8)
        	{
        		nrOfWrittenBytes++;
        		out.write(carry);
        		nrOfCarried = 0;
        		carry = 0;
        	}
        }
        
        if (nrOfCarried != 0)
        {
        	nrOfWrittenBytes++;
        	out.write(carry);
    		nrOfCarried = 0;
    		carry = 0;
        }
        
        return nrOfWrittenBytes;
    }
    
    /**
     * Returns reversed byte array holding bytes written to output buffer and shifts
     * coded data so that it begins at the beginning of the first byte 
     * @return
     */
    private byte[] getOutputByteArray()
    {
    	byte[] arrRev = new byte[outBuffer.size()];
        outBuffer.toByteArray(0, arrRev.length, arrRev, 0);
        
        int rb = nrOfBits % 8;
    	int eb = 8 - rb;
    	int rbMask = (1 << rb) - 1;
    	int ebMask = (1 << eb) - 1;
        byte[] result = new byte[arrRev.length];
        if (rb != 0) // reverse the byte array and shift to the beginning
        {
        	for (int i = arrRev.length - 1; i >= 1; i--)
            {
            	int front = arrRev[i] & rbMask;
            	int back = arrRev[i - 1] & (ebMask << rb);
            	result[arrRev.length - 1 - i] = (byte) ((front << eb) | (back >>> rb));
            }
            
            result[result.length - 1] = (byte) ((arrRev[0] & rbMask) << eb);
        }
        else // no shift is required
        {
        	for (int i = arrRev.length - 1; i >= 0; i--)
            {
            	result[arrRev.length - 1 - i] = arrRev[i];
            }
        }
        
        return result;
    }
}
