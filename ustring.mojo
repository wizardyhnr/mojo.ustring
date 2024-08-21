from algorithm.functional import vectorize
from memory.unsafe import DTypePointer
from memory.unsafe_pointer import UnsafePointer
from sys.info import simdwidthof
from collections import List
from prefix_sum import simd_prefix_sum, simd_idx_sum

alias simd_width_u8 = simdwidthof[DType.uint8]()

struct UString(Sized, Boolable, Stringable, Copyable, Movable):
    # String encoded in utf-8.
    var inner_string: String
    # char length at each char index
    # [3, 1, 2, ...] for example
    # length of idx equals to length of chars.
    var idx: List[UInt8]
    # length of chars in inner_string
    var length: Int

    """
    To save space, length of a utf-8 encoded char 
    could be from 1 to 4. 2bit data 00(len=1), 01(len=2),
    10(len=3), 11(len=4).
    An element of idx can hold length info of 4 chars.
    One issue is not all bits of UInt8 may be used up, 
    for last element of idx, needs to check out of boundary.
    """

    """
	Features of utf-8 coding:
    first code point,	Last code point,	Byte1,		Byte2,		Byte3,		Byte4
    U+0000,				U+007F,				0xxxxxxx, 	NA,			NA,			NA
    U+0080,				U+07FF,				110xxxxx,	10xxxxxx,	NA,			NA
    U+0800,				U+FFFF,				1110xxxx,	10xxxxxx,	10xxxxxx,	NA
    U+10000,			U+10FFFF,			11110xxx,	10xxxxxx,	10xxxxxx,	10xxxxxx.
    """

    fn __init__(inout self, owned input: String):
        self.inner_string = input
        self.idx = List[UInt8]()
        self.length = 0
        self._count_chars()
        self.idx = List[UInt8](capacity=self.length)

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return len(self) > 0

    @always_inline
    fn __str__(self) -> String:
        return str(self.inner_string)

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        """Creates a deep copy of an existing string.

        Args:
            existing: The string to copy.
        """
        self.inner_string = existing.inner_string
        self.idx = existing.idx
        self.length = existing.length

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        """Move the value of a string.

        Args:
            existing: The string to move.
        """
        self.inner_string = existing.inner_string^
        self.idx = existing.idx^
        self.length = existing.length

    # build initial Unicode index from a utf-8 encoded Str
    fn _build_idx(inout self):
        var p = DTypePointer(self.inner_string.unsafe_uint8_ptr())
        var str_byte_len = len(self.inner_string)
        var result = DTypePointer[DType.uint8].alloc(str_byte_len)
        var dptr_cache = DTypePointer[DType.uint8].alloc(len(self))

        # fill self.idx with code point width of each char.
        @parameter
        fn calc_char_width[simd_width: Int](offset: Int):
            var ret_simd = (countl_zero[DType.uint8, simd_width](\
                                p.load[width=simd_width](offset) ^ 0b11110000)\
                   )
            #print("simd width is:", simd_width)
            result.store[width=simd_width](offset, ret_simd)

        vectorize[calc_char_width, simd_width_u8](str_byte_len)
        #calc_char_width[simd_width_u8](0)
        self.idx.clear()
        var char_idx = 0
        for i in range(str_byte_len):
            if result[i] != 1:
                if result[i] == 0:
                    dptr_cache[char_idx] = 1
                    #print(1)
                else:
                    dptr_cache[char_idx] = result[i]
                    #print(result[i])
                char_idx = char_idx + 1
        
        #print("print wc_cache\n========")
        #for i in range(len(self)):
        #    print(dptr_cache[i])
        # assuming all chars are 4 bytes.
        # length of 64 ensures accumlated index does not exceed 255
        # to avoid overflow of uint8
        var width = 64
        var start = 0 
        while start + width <= len(self):
#for i in range(start, start+width+1):
#                print(i, dptr_cache[i])
#            print("======")
            simd_idx_sum[DType.uint8](dptr_cache.offset(start), size=width)
            #print("start pos: ", start)
#            for i in range(start, start+width+1):
#                print(i, dptr_cache[i])
#            print("======")
            start = start+width
        var leftover = len(self) - start
        if leftover > 0:
            #print("start pos: ", start)
#            for i in range(start, start+leftover+1):
#                print(i, dptr_cache[i])
#            print("======")

            simd_idx_sum[DType.uint8](dptr_cache.offset(start), size=leftover)
#            for i in range(start, start+leftover+1):
#                print(i, dptr_cache[i])
            #print("print wc_cache\n====")
            #for i in range(len(self)):
                #print(dptr_cache[i])

        for i in range(len(self)):
            self.idx.append(dptr_cache[i])

        result.free()
        dptr_cache.free()

    fn _calc_slice(self, char_idx: Int) -> Slice:
        var p = DTypePointer(self.idx.unsafe_ptr())
        var str_char_len = len(self)
        var start = 0
        var end = 0
        if char_idx < 0:
            return self._calc_slice(str_char_len + char_idx)

        debug_assert((0 <= char_idx < str_char_len), "index must be in range")
 
        if char_idx == 0:
            return Slice(0, int(p[0])+1)
        var m = (char_idx-1) // 64
        var n = (char_idx) // 64
        for i in range(m):
            start = start + int(p[64*m-1])
        for i in range(n):
            end = end + int(p[64*n-1])
        return Slice(start+int(p[char_idx-1])+1, end+int(p[char_idx])+1)

    # return number of Unicode chars from a utf-8 encoded Str
    # count occurancies of bytes not starting with 0b10.
    fn _count_chars(inout self):
        var p = DTypePointer(self.inner_string.unsafe_uint8_ptr())
        var str_byte_len = len(self.inner_string)
        var result = 0

        @parameter
        fn count[simd_width: Int](offset: Int):
            result += int( \
					  ((p.load[width=simd_width](offset) >> 6) != 0b10) \
					  .cast[DType.uint8]() \
			          .reduce_add() \
                      )

        vectorize[count, simd_width_u8](str_byte_len)
        self.length = result

    fn __len__(self) -> Int:
        return self.length

	# when underlying utf-8 encoded Str is changed, call \
	# this method to update len and index as well.
	
    fn _update_idx(inout self):
        self._count_chars()
        self._build_idx()

    fn __getitem__(self, idx: Int) -> String:
        """Gets the character at the specified position.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        if idx < 0:
            return self.__getitem__(len(self) + idx)

        debug_assert(0 <= idx < len(self), "index must be in range")
        var buf = List[UInt8](capacity=4)
        
        var slc = self._calc_slice(idx)
        return self.inner_string[slc]

    fn __getitem__(self, span: Slice) -> String:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new string containing the string at the specified positions.
        """

        var adjusted_span = self._adjust_span(span)
        var adjusted_span_len = adjusted_span.unsafe_indices()
#print("span, start", adjusted_span.start, "end", adjusted_span.end, "step", adjusted_span.step, "len", adjusted_span_len)
        var byte_start = self._calc_slice(adjusted_span.start).start
        var byte_end = self._calc_slice(adjusted_span.end-1).end 
#print("byte span, start", byte_start, "end", byte_end)
        if adjusted_span.step == 1:
            return self.inner_string[byte_start:byte_end]

        var buffer: String = ""
        for i in range(adjusted_span_len):
            var slc = self._calc_slice(adjusted_span[i])
            buffer += self.inner_string[slc]
        return buffer^

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the string length."""
        var adjusted_span = span

        if adjusted_span.start < 0:
            adjusted_span.start = len(self) + adjusted_span.start

        if not adjusted_span._has_end():
            adjusted_span.end = len(self)
        elif adjusted_span.end < 0:
            adjusted_span.end = len(self) + adjusted_span.end

        if span.step < 0:
            var tmp = adjusted_span.end
            adjusted_span.end = adjusted_span.start - 1
            adjusted_span.start = tmp - 1

        return adjusted_span


