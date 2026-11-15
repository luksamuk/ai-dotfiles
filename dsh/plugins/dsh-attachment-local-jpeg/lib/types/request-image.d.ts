/** Deterministic cached image versions for model requests. */
import { ImageVariantId } from '@deepseek-ai/dsh-attachment';
import type { ImageAttachmentRef, ImageRequestTarget, RequestImageAttachment, StoredImageAttachment } from '@deepseek-ai/dsh-attachment';
/** Transform version included in every cache and upload-index identity. */
export declare const REQUEST_IMAGE_TRANSFORM_VERSION = "request-image-v6";
/**
 * Complete deterministic identity for one attachment and route-chosen request target.
 * @param attachment - provider-independent durable normalized attachment reference.
 * @param target - route-chosen dimensions and byte target.
 * @returns branded digest over every request transform input.
 */
export declare function requestImageVariantId(attachment: ImageAttachmentRef, target: ImageRequestTarget): ReturnType<typeof ImageVariantId>;
/**
 * Generate or reuse one request image below the local attachment cache root.
 * @param root - absolute attachment cache root; variants use its `request-images` child.
 * @param attachment - verified normalized attachment bytes and reference.
 * @param target - exact route-chosen dimensions and byte target; a target above the source keeps the source size.
 * @param signal - optional cancellation for cache I/O and image transformation.
 * @returns verified request bytes and deterministic variant identity.
 */
export declare function readRequestImageFile(root: string, attachment: StoredImageAttachment, target: ImageRequestTarget, signal?: AbortSignal): Promise<RequestImageAttachment>;
//# sourceMappingURL=request-image.d.ts.map