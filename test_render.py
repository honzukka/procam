import torch
import pytest

import render


image_width = 640
image_height = 640
basis_size = 400
batch_size = 20
iterations = 1
seed = 1234


def ref_matmul(h, w, b, n):
    torch.manual_seed(seed)
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    matrix = torch.rand((h, w, 3, b), dtype=torch.float32)
    images = torch.rand((n, b, 3), dtype=torch.float32)

    matrix = matrix.to(device)
    images = images.to(device)

    return torch.einsum('hwcb,nbc->nhwc', matrix, images).squeeze()


class TestMatmul:
    def test_matmul_ref(self):
        h = 48
        w = 64
        b = 100
        n = 10

        torch.manual_seed(seed)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        matrix = torch.rand((h, w, 3, b), dtype=torch.float32)
        images = torch.rand((n, b, 3), dtype=torch.float32)

        matrix = matrix.to(device)
        images = images.to(device)

        actual = render.project(images, matrix)
        expected = ref_matmul(h, w, b, n)

        assert torch.equal(actual, expected)

    def test_matmul_mm(self):
        h = 48
        w = 64
        b = 100
        n = 1

        torch.manual_seed(seed)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        matrix = torch.rand((h * w * 3, b), dtype=torch.float32)
        images = torch.rand((b, 3), dtype=torch.float32)

        matrix = matrix.to(device)
        images = images.to(device)

        actual = (
            torch.mm(matrix, images)
            .reshape(h, w, 3, 3)
            .diagonal(dim1=2, dim2=3)
        )
        expected = ref_matmul(h, w, b, n)

        assert torch.allclose(actual, expected)

    def test_matmul_matmul(self):
        h = 48
        w = 64
        b = 100
        n = 10

        torch.manual_seed(seed)
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        matrix = torch.rand((h * w * 3, b), dtype=torch.float32)
        images = torch.rand((n, b, 3), dtype=torch.float32)

        matrix = matrix.to(device)
        images = images.to(device)

        actual = (
            torch.matmul(matrix, images)
            .reshape(n, h, w, 3, 3)
            .diagonal(dim1=3, dim2=4)
        )
        expected = ref_matmul(h, w, b, n)

        assert torch.allclose(actual, expected)


@pytest.mark.benchmark
class TestBenchmarkMatmul:
    def matmul_pbc(self, matrix, images, h, w):
        result = None
        for i in range(iterations):
            result = torch.einsum(
                'pbc,nbc->npc', matrix, images
            ).reshape(-1, h, w, 3)
        return result

    def matmul_bhwc(self, matrix, images):
        result = None
        for i in range(iterations):
            result = torch.einsum('bhwc,nbc->nhwc', matrix, images)
        return result

    def matmul_bpc(self, matrix, images, h, w):
        result = None
        for i in range(iterations):
            result = torch.einsum(
                'bpc,nbc->npc', matrix, images
            ).reshape(-1, h, w, 3)
        return result

    def matmul_hwbc(self, matrix, images):
        result = None
        for i in range(iterations):
            result = torch.einsum('hwbc,nbc->nhwc', matrix, images)
        return result

    def matmul_chwb(self, matrix, images):
        result = None
        for i in range(iterations):
            result = torch.einsum('chwb,nbc->nhwc', matrix, images)
        return result

    def matmul_bchw(self, matrix, images):
        result = None
        for i in range(iterations):
            result = torch.einsum('bchw,nbc->nhwc', matrix, images)
        return result

    def matmul_hwcb(self, matrix, images):
        result = None
        for i in range(iterations):
            result = torch.einsum('hwcb,nbc->nhwc', matrix, images)
        return result

    def matmul_cpb(self, matrix, images, h, w):
        result = None
        for i in range(iterations):
            result = torch.einsum(
                'cpb,nbc->npc', matrix, images
            ).reshape(-1, h, w, 3)
        return result

    def matmul_pcb(self, matrix, images, h, w):
        result = None
        for i in range(iterations):
            result = torch.einsum(
                'pcb,nbc->npc', matrix, images
            ).reshape(-1, h, w, 3)
        return result

    def matmul_mm(self, matrix, images, h, w):
        result = None
        for i in range(iterations):
            result = (
                torch.mm(matrix, images)
                .reshape(h, w, 3, 3)
                .diagonal(dim1=2, dim2=3)
            )
        return result

    def matmul_matmul(self, matrix, images, h, w, n):
        result = None
        for i in range(iterations):
            result = (
                torch.matmul(matrix, images)
                .reshape(n, h, w, 3, 3)
                .diagonal(dim1=3, dim2=4)
            )
        return result

    def test_matmul_pbc(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_width * image_height, basis_size, 3
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_pbc, matrix, images,
            image_height, image_width
        )
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_bhwc(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            basis_size, image_height, image_width, 3
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(self.matmul_bhwc, matrix, images)
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_bpc(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            basis_size, image_height * image_width, 3
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_bpc, matrix, images, image_height, image_width
        )
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_hwbc(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_height, image_width, basis_size, 3
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(self.matmul_hwbc, matrix, images)
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_chwb(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            3, image_height, image_width, basis_size
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(self.matmul_chwb, matrix, images)
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_bchw(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            basis_size, 3, image_height, image_width
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(self.matmul_bchw, matrix, images)
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_hwcb(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_height, image_width, 3, basis_size
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(self.matmul_hwcb, matrix, images)
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_cpb(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            3, image_height * image_width, basis_size
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_cpb, matrix, images, image_height, image_width
        )
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_pcb(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_height * image_width, 3, basis_size
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_pcb, matrix, images, image_height, image_width
        )
        assert result.shape == (batch_size, image_height, image_width, 3)

    def test_matmul_mm(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_height * image_width * 3, basis_size
        )
        images = torch.randn(basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_mm, matrix, images, image_height, image_width
        )
        assert result.shape == (image_height, image_width, 3)

    def test_matmul_matmul(self, benchmark):
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        matrix = torch.randn(
            image_height * image_width * 3, basis_size
        )
        images = torch.randn(batch_size, basis_size, 3)

        matrix.to(device)
        images.to(device)

        result = benchmark(
            self.matmul_matmul, matrix, images,
            image_height, image_width, batch_size
        )
        assert result.shape == (batch_size, image_height, image_width, 3)


class BenchmarkMatmulCUDA:
    def __init__(self, iters=100, n=1, b=64*40, h=480, w=640):
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA not available, cannot benchmark CUDA functions!'
            )

        self.iters = iters
        self.device = torch.device('cuda')

        self.n = n
        self.b = b
        self.h = h
        self.w = w

    def einsum_hwcb(self):
        matrix = torch.randn(
            (self.h, self.w, 3, self.b),
            dtype=torch.float32, device=self.device
        )
        texture = torch.randn(
            (self.b, 3),
            dtype=torch.float32, device=self.device
        )
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        elapsed_time_ms = 0.0
        for i in range(self.iters):
            start_event.record()
            result = torch.einsum('hwcb,bc->hwc', matrix, texture)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

        print('einsum_hwcb():')
        print('{:.4f}ms / iteration'.format(elapsed_time_ms / self.iters))
        print('Result shape: {}'.format(result.shape))

    def einsum_chwb(self):
        matrix = torch.randn(
            (3, self.h, self.w, self.b),
            dtype=torch.float32, device=self.device
        )
        texture = torch.randn(
            (3, self.b),
            dtype=torch.float32, device=self.device
        )
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        elapsed_time_ms = 0.0
        for i in range(self.iters):
            start_event.record()
            result = torch.einsum('chwb,cb->hwc', matrix, texture)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

        print('einsum_chwb():')
        print('{:.4f}ms / iteration'.format(elapsed_time_ms / self.iters))
        print('Result shape: {}'.format(result.shape))

    def matmul_chwb(self):
        matrix = torch.randn(
            (3, self.h * self.w, self.b),
            dtype=torch.float32, device=self.device
        )
        texture = torch.randn(
            (3, self.b, 1),
            dtype=torch.float32, device=self.device
        )
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        elapsed_time_ms = 0.0
        for i in range(self.iters):
            start_event.record()
            result = torch.matmul(
                matrix,
                texture
            )
            end_event.record()
            torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

        print('matmul_chwb():')
        print('{:.4f}ms / iteration'.format(elapsed_time_ms / self.iters))
        print('Result shape: {}'.format(result.shape))
