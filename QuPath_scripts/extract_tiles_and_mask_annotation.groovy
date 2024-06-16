import qupath.lib.images.servers.LabeledImageServer

// Variables necessary for the internal operation of the program
def imageData = getCurrentImageData()
def server = imageData.getServer()

// WSI bounds (offset that has to be applied to get the correct coordinates in OpenSlide)
double dx = -getCurrentServer().boundsX
double dy = -getCurrentServer().boundsY
print 'Bounds WSI: dx = ' + dx + ', dy = ' + dy

// Path to save the extracted tiles
def pathOutput = "output_path" // 
mkdirs(pathOutput)

// Name of the original image (WSI), identifier
def name = GeneralTools.getNameWithoutExtension(server.getMetadata().getName())

// Define output downsample: 40x: 1.0, 20x: 2.0, 10x: 4.0, 5x: 8.0
double downsample = 4.0
def size = 256
def overlap_pixels = 0

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
                  .downsample(downsample)                                         // Choose server resolution; this should match the resolution at which tiles are exported
                  
                  .backgroundLabel(0, ColorTools.BLACK)                           // Specify background label
                  // .addLabel('Other', 1, ColorTools.makeRGB(85, 85, 85))           // Classes labels (the order matters!)
                  // .addLabel('Region to annotate', 0, ColorTools.makeRGB(0, 0, 0))
                  .addLabel('tissue', 1, ColorTools.makeRGB(85,85,85))
                  .addLabel('Invasive', 2, ColorTools.makeRGB(170, 170, 170))
                  .addLabel('In situ', 3, ColorTools.makeRGB(255, 255, 255))
                  
                  .multichannelOutput(false)                                      // If true, each label is a different channel (required for multiclass probability)
                  .build()


// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)                                         // Define export resolution
    .imageExtension('.png')                                         // Define file extension for original pixels, the image
    // .labeledImageExtension('.png')
    
    .imageSubDir('Images')                                          // Subdirectory to save the original images
    .labeledImageSubDir('Masks')                                    // Subdirectory to save the masks
    
    .tileSize(size)                                                 // Define size of each tile, in pixels
    .labeledServer(labelServer)                                     // Define the labeled image server to use (i.e. the one we just built)
    // .annotatedTilesOnly(true)                                    // If true, only export tiles if there is a (labeled) annotation present
    .annotatedCentroidTilesOnly(true)                               // If true, only export tiles if the centroid pixels has an annotation
    .overlap(overlap_pixels)                                        // Define overlap, in pixel units at the EXPORT resolution
    .writeTiles(pathOutput)                                         // Write tiles to the specified directory
    
print 'Done!'