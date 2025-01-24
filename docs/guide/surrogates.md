# Surrogates

A surrogate model can make the optimization more efficient by building an approximate model of the problem that can be queried at a faster rate. The most promising points according to the surrogate model can then be evaluated at the actual problem. dmosopt implements various strategies listed below:

<ul>
    <li v-for="i in ['gpr', 'egp', 'megp', 'mdgp', 'mdspp','vgp', 'svgp', 'spv', 'siv', 'crv']">
        {{ i }} - <a href="https://github.com/iraikov/dmosopt/blob/main/dmosopt/model.py" target="_blank">
            {{ i.toUpperCase() }}_Matern
        </a>
    </li>
</ul>

You may also point to your custom implementations by specifying a Python import path.
