from utils.loop import unroll
from sys.info import simdwidthof, simdbitwidth
from collections import List

alias simd_width_b8 = simdwidthof[DType.uint8]()
alias simd_width_b16 = simdwidthof[DType.uint16]()
alias simd_width_b32 = simdwidthof[DType.uint32]()
alias simd_width_b64 = simdwidthof[DType.uint64]()

@always_inline
fn scalar_prefix_sum[D: DType](inout array: List[SIMD[D, 1]]):
    var element = array[0]
    for i in range(1, len(array)):
        array[i] += element
        element = array[i]

# for utf-8 encoding, max bytes per char are 4.
# To use a uint8 to record byte number,
# it can cover at least 256/4=64 chars.
# So a List[UInt8](capacity=64) can record index
# of 64 chars without any problem.

# prefix sum calculation on one chunk.
# data should be split into chunks with simd_width_dtype
# and last chunk is leftover.
@always_inline
fn _sum[width: Int, loops: Int, D: DType](pointer: DTypePointer[D], carry_over: SIMD[D, 1]) -> SIMD[D, width]:
    # width: simd width based on D.
    # aligment not used, to delete
    # loops shift number to be used, needs to be positive.
    # caller to deal with loops = 0 case.

    # relationship between width and loops:
    # 64 -> 6
    # 32 -> 5
    # 16 -> 4
    # 8 -> 3
    # 4 -> 2
    # 2 -> 1
    # 1 -> 0, this does not need unroll.
    # for 256 SIMD, width of uint8 is 32.
    var result = pointer.load[width=width]()
    
    #for simd width of 32, shift order is 1, 2, 4, 8, 16.
    #as a result, loops is between 0, log2(32).
    #so that 1<<loops generates 1, 2, 4, 8, 16.
    @parameter
    fn add[i: Int]():
        result += result.shift_right[1 << i]()
    
    # unroll add in 0, 1, ..., loops-1.
    # for width=32, loops=5
    # shift_right(1)
    # shift_right(2)
    # ...
    # shift_right(16)
    unroll[add, loops]()
    # carry_over is last element of last chunk.
    # carry_over is casted to SIMD[DType, simd_width_DType]
    result += carry_over
    return result

@always_inline
fn simd_prefix_sum[D: DType](inout array: List[SIMD[D, 1]]):
    # split array into chunks with size of simd_width_DType and leftover.
    # width is simd_width_DType
    @parameter 
    fn inner_func[width: Int, loops: Int]():
        var length = len(array)
        var numbers = DTypePointer[D](array.unsafe_ptr())
        # carry_over
        var c: SIMD[D, 1] = 0
        # index of chunk
        var i = 0

        while i + width <= length:
            var part = _sum[width, loops, D](numbers.offset(i), c)
            # get carry_over value from last element of last chunk.
            c = part[width - 1]
            # store finished chunk
            numbers.store(i, part)
            i += width

        # leftover <= chunk size.
        # because width of SIMD needs to be power of 2.
        # add_rest change width of SIMD from width/2, width/4 to 2.
        @parameter
        fn add_rest[round: Int]():
            alias index = round + 1
            # width >> index detects leftover's size in simd_width_DType/2,
            # simd_width_DType/4, ..., 1 like biarny seach.
            alias w = width >> index
            if i + w <= length:
                var part = _sum[w, loops - index, D](numbers.offset(i), c)
                c = part[w - 1]
                numbers.store(i, part)
                i += w    

        # unroll add_rest in ranger(loops), process SIMD width down to 2.
        unroll[add_rest, loops-1]()
        
        # last element is SIMD[DType, 1], length is odd number.
        if (i+1) == length:
            numbers.store(i, numbers.offset(i).load[width=1]() + c)

    @parameter
    if simdbitwidth() == 256:
        if D == DType.uint32 or D == DType.int32 or D == DType.float32:
            # for 256bit SIMD, simd_width_b32 is 8, 0b1000
            # loops = 3
            alias loops = 3
            # shift 1bit, 2bit, 4bit
            inner_func[simd_width_b32, loops]()
        elif D == DType.uint16 or D == DType.int16 or D == DType.float16:
            # for 256bit SIMD, simd_width_b16 is 16, 0b10000
            alias loops = 4
            # shift 1bit, 2bit, 4bit, 8bit       
            inner_func[simd_width_b16, loops]()
        elif D == DType.uint8 or D == DType.int8:
            # for 256bit SIMD, simd_width_b8 is 32, 0b100000
            # get log2(loops) by couting trailing zeros.
            alias loops = 5
            # shift 1bit, 2bit, 4bit, 8bit, 16bit
            inner_func[simd_width_b8, loops]()
        else:
            # DType.64bit
            # for 256bit, SIMD, simd_width_b64 is 4, 0b100
            var width = simd_width_b64
            alias loops = 2
            # shift 1bit
            inner_func[simd_width_b64, loops]()

    if simdbitwidth() == 512:
        if D == DType.uint32 or D == DType.int32 or D == DType.float32:
            # for 512bit SIMD, simd_width_b32 is 16, 0b10000
            # loops = 4
            alias loops = 4
            # shift 1bit, 2bit, 4bit, 8bit
            inner_func[simd_width_b32, loops]()
        elif D == DType.uint16 or D == DType.int16 or D == DType.float16:
            # for 512bit SIMD, simd_width_b16 is 32, 0b100000
            alias loops = 5
            # shift 1bit, 2bit, 4bit, 8bit, 16bit       
            inner_func[simd_width_b16, loops]()
        elif D == DType.uint8 or D == DType.int8:
            # for 512bit SIMD, simd_width_b8 is 64, 0b1000000
            # get log2(loops) by couting trailing zeros.
            alias loops = 6
            # shift 1bit, 2bit, 4bit, 8bit, 16bit, 32bit
            inner_func[simd_width_b8, loops]()
        else:
            # DType.64bit
            # for 512bit, SIMD, simd_width_b64 is 8, 0b1000
            alias loops = 3
            # shift 1bit, 2bit, 4bit
            inner_func[simd_width_b64, loops]()

@always_inline
fn simd_idx_sum[D: DType](array: DTypePointer[D], size: Int):
    # split array into chunks with size of simd_width_DType and leftover.
    # width is simd_width_DType
    @parameter 
    fn inner_func[width: Int, loops: Int]():
        var length = size
        var numbers = array
        # carry_over
        var c: SIMD[D, 1] = 255
        # index of chunk
        var i = 0

        while i + width <= length:
#print("carry_over is:", c)
#            for n in range(width+1):
#                print(n, numbers.offset(i)[n])

            var part = _sum[width, loops, D](numbers.offset(i), c)
            # get carry_over value from last element of last chunk.
            c = part[i+width - 1]
            # store finished chunk
#            print("***")
#            print(part)
            numbers.store(i, part)
#            for n in range(width+1):
#                print(n, numbers.offset(i)[n])
#            print("continue")

            i += width

        # leftover <= chunk size.
        # because width of SIMD needs to be power of 2.
        # so uses binary search method to spilt leftover until 1 or 0.
        @parameter
        fn add_rest[round: Int]():
            alias index = round + 1
            # width >> index detects leftover's size in simd_width_DType/2,
            # simd_width_DType/4, ..., 1 like biarny seach.
            alias w = width >> index
            if i + w <= length:
                var part = _sum[w, loops - index, D](numbers.offset(i), c)
                c = part[w - 1]
                numbers.store(i, part)
                i += w    

        # unroll add_rest in ranger(loops), process SIMD width down to 2.
        unroll[add_rest, loops-1]()
        if (i+1) == length:
            # last element is SIMD[DType, 1], length is odd number.
            numbers.store(i, numbers.offset(i).load[width=1]() + c)

    @parameter
    if simdbitwidth() == 256:
        if D == DType.uint32 or D == DType.int32 or D == DType.float32:
            # for 256bit SIMD, simd_width_b32 is 8, 0b1000
            # loops = 3
            alias loops = 3
            # shift 1bit, 2bit, 4bit
            inner_func[simd_width_b32, loops]()
        elif D == DType.uint16 or D == DType.int16 or D == DType.float16:
            # for 256bit SIMD, simd_width_b16 is 16, 0b10000
            alias loops = 4
            # shift 1bit, 2bit, 4bit, 8bit       
            inner_func[simd_width_b16, loops]()
        elif D == DType.uint8 or D == DType.int8:
            # for 256bit SIMD, simd_width_b8 is 32, 0b100000
            # get log2(loops) by couting trailing zeros.
            alias loops = 5
            # shift 1bit, 2bit, 4bit, 8bit, 16bit
            inner_func[simd_width_b8, loops]()
        else:
            # DType.64bit
            # for 256bit, SIMD, simd_width_b64 is 4, 0b100
            var width = simd_width_b64
            alias loops = 2
            # shift 1bit
            inner_func[simd_width_b64, loops]()

    if simdbitwidth() == 512:
        if D == DType.uint32 or D == DType.int32 or D == DType.float32:
            # for 512bit SIMD, simd_width_b32 is 16, 0b10000
            # loops = 4
            alias loops = 4
            # shift 1bit, 2bit, 4bit, 8bit
            inner_func[simd_width_b32, loops]()
        elif D == DType.uint16 or D == DType.int16 or D == DType.float16:
            # for 512bit SIMD, simd_width_b16 is 32, 0b100000
            alias loops = 5
            # shift 1bit, 2bit, 4bit, 8bit, 16bit       
            inner_func[simd_width_b16, loops]()
        elif D == DType.uint8 or D == DType.int8:
            # for 512bit SIMD, simd_width_b8 is 64, 0b1000000
            # get log2(loops) by couting trailing zeros.
            alias loops = 6
            # shift 1bit, 2bit, 4bit, 8bit, 16bit, 32bit
            inner_func[simd_width_b8, loops]()
        else:
            # DType.64bit
            # for 512bit, SIMD, simd_width_b64 is 8, 0b1000
            alias loops = 3
            # shift 1bit, 2bit, 4bit
            inner_func[simd_width_b64, loops]()


