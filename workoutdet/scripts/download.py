import base64
from os.path import join as osj


def parse_onedrive(link: str) -> str:
    """Parse onedrive link to download link.

    Args:
        link (str): start with `https://1drv.ms/u/s!`
    Returns:
        str, download link.
    """
    assert link.startswith('https://1drv.ms/u/s!')
    b = base64.urlsafe_b64encode(link.strip().encode('ascii'))
    s = b.decode('ascii')
    res = f'https://api.onedrive.com/v1.0/shares/u!{s}/root/content'
    return res


class Downloader:
    _BASE = 'https://1drv.ms/u/s!AiohV3HRf'
    _DATASETS = {
        'repcountA': '34iptWkA3RXj6Y_wspxw',
        'repcount_anno': "34i_YvMob5Vpgvxjc3mQ",
        'repcount_rawframes': "34ipwACYfKSHhkZzebrQ",
        'countix': "34ipwWwzztzGynyj5Fwg",
    }
    _WEIGHTS = {
        'ssv2_videomae_pretrain_base_patch16_224_frame_16x2_tube_mask_ratio_0.9_e2400.pth':
            '34jJNTPAlz0XrR9CQM_A',
        'TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e45.pth':
            '34jIdNWFvaFEZb6BpC6g',
        'tdn_sthv2_r50_8x1x1.pth':
            '34jIdRXV1Kogylklw5lA'
    }

    def __init__(self, root: str):
        self.root = root

    def get_dataset(self, name: str) -> str:
        link = f'{self._BASE}-{self._DATASETS[name]}'
        return parse_onedrive(link)

    def get_weight(self, name: str) -> str:
        link = f'{self._BASE}-{self._WEIGHTS[name]}'
        return parse_onedrive(link)

    @property
    def datasets(self):
        return list(self._DATASETS.keys())

    @property
    def weights(self):
        return list(self._WEIGHTS.keys())

    def download(self, name: str, path: str):
        pass

    def _download_repcount(self, path: str) -> str:
        """Returns download and extraction shell command."""
        link = self.get_dataset('repcountA')
        cmd = [
            f'wget -O /tmp/content.tar {link}', f'tar xf /tmp/content.tar /tmp/',
            f'rm /tmp/content.tar', f'mv /tmp/LLSP/video {osj(path, "videos")}',
            f'mv {osj(path, "videos/valid")} {osj(path, "videos/val")}'
        ]
        return '\n'.join(cmd)

    def __repr__(self):
        return f'Downloader({self.root})'


if __name__ == '__main__':
    downloader = Downloader('checkpoints')
    print(downloader.weights)
