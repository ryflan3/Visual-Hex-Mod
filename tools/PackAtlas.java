import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.tools.texturepacker.TexturePacker;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/**
 * Packs the mod's Images folder into game.atlas + game.png the same way Unciv's desktop
 * packer does (desktop/src/com/unciv/app/desktop/ImagePacker.kt), so Android players get
 * the same result as desktop players. Android can't pack images itself.
 *
 * Usage (from the repo root, with the libGDX jars on the classpath):
 *   java -cp gdx.jar:gdx-tools.jar tools/PackAtlas.java .
 */
public class PackAtlas {
    public static void main(String[] args) throws Exception {
        File root = new File(args.length > 0 ? args[0] : ".").getCanonicalFile();
        File images = new File(root, "Images");
        if (!images.isDirectory()) throw new IllegalStateException("No Images folder in " + root);

        TexturePacker.Settings settings = new TexturePacker.Settings();
        // Some mobile GPUs only support 2048x2048 textures, so Unciv caps pages at this size.
        settings.maxWidth = 2048;
        settings.maxHeight = 2048;
        settings.combineSubdirectories = true;
        settings.pot = true;
        settings.fast = true;
        // Padding plus duplicatePadding stop neighbouring images bleeding into each other in mipmaps.
        settings.paddingX = 8;
        settings.paddingY = 8;
        settings.duplicatePadding = true;
        settings.filterMin = Texture.TextureFilter.MipMapLinearLinear;
        settings.filterMag = Texture.TextureFilter.MipMapLinearLinear;

        for (File old : root.listFiles()) {
            String n = old.getName();
            if (n.equals("game.atlas") || (n.startsWith("game") && n.endsWith(".png"))) old.delete();
        }
        TexturePacker.process(settings, images.getPath(), root.getPath(), "game");
        Files.write(new File(root, "Atlases.json").toPath(), "[game]".getBytes(StandardCharsets.UTF_8));
    }
}
